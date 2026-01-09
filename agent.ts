import { Annotation, StateGraph, END, START } from "@langchain/langgraph";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { ChatOllama } from "@langchain/ollama";
import { createClient } from "@supabase/supabase-js";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import pg from "pg";
import {
    SUPABASE_URL,
    SUPABASE_PRIVATE_KEY,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    DATABASE_URL
} from "./config.ts";

const { Pool } = pg;

// 1. Enhanced Graph State with Citations and Persona
const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    question: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    route: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    documents: Annotation<Document[]>({
        reducer: (x, y) => (y ? (x ?? []).concat(y) : x),
        default: () => [],
    }),
    compressedContext: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    generation: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    citations: Annotation<string[]>({
        reducer: (x, y) => y ?? x,
        default: () => [],
    }),
    userPersona: Annotation<any>({
        reducer: (x, y) => y ?? x,
        default: () => ({}),
    }),
    steps: Annotation<string[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    loopCount: Annotation<number>({
        reducer: (x, y) => y ?? x,
        default: () => 0,
    }),
    hallucinationCheck: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    answerCheck: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
});

// 2. Setup Clients
const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY);
const pool = new Pool({ connectionString: DATABASE_URL });
const checkpointer = new PostgresSaver(pool);

const embeddings = {
    embedQuery: async (text: any) => {
        const queryText = typeof text === "string" ? text : (text.input || text.pageContent || JSON.stringify(text));
        const res = await fetch(`${OLLAMA_BASE_URL}/api/embed`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: EMBEDDING_MODEL, input: [queryText] }),
        });
        if (!res.ok) throw new Error(`Ollama error: ${await res.text()}`);
        const data = await res.json();
        return data.embeddings[0];
    },
    embedDocuments: async (texts: string[]) => {
        const res = await fetch(`${OLLAMA_BASE_URL}/api/embed`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: EMBEDDING_MODEL, input: texts }),
        });
        if (!res.ok) throw new Error(`Ollama error: ${await res.text()}`);
        const data = await res.json();
        return data.embeddings;
    }
} as any;

const vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabase,
    tableName: "documents",
    queryName: "match_documents",
});

const llm = new ChatOllama({
    baseUrl: OLLAMA_BASE_URL,
    model: LLM_MODEL,
    temperature: 0,
});

// 3. Helper Functions
function extractContent(content: any): string {
    if (typeof content === "string") return content;
    if (content?.text) return content.text;
    if (content?.input) return content.input;
    if (Array.isArray(content)) {
        return content.map((c: any) => typeof c === "string" ? c : (c.text || JSON.stringify(c))).join(" ");
    }
    return JSON.stringify(content);
}

// IMPROVEMENT 2: User Persona Management
async function loadUserPersona(userId: string = "default") {
    const { data, error } = await supabase
        .from("user_personas")
        .select("persona_data")
        .eq("user_id", userId)
        .single();

    if (error || !data) return {};
    return data.persona_data || {};
}

async function updateUserPersona(userId: string = "default", updates: any) {
    const { error } = await supabase
        .from("user_personas")
        .upsert({
            user_id: userId,
            persona_data: updates,
            updated_at: new Date().toISOString()
        });

    if (error) console.error("Failed to update persona:", error);
}

// 4. Nodes

async function routeQuery(state: typeof GraphState.State) {
    console.log("--- ROUTING QUERY ---");
    const lastMessage = state.messages[state.messages.length - 1];
    const question = extractContent(lastMessage.content);

    // Load user persona
    const persona = await loadUserPersona();

    const routerPrompt = ChatPromptTemplate.fromTemplate(`
    Route the user question.
    
    User Profile: {persona}
    Question: {question}
    
    Paths:
    - 'vectorstore': Technical questions about Bitcoin, blockchain, or PDF content
    - 'conversational': Greetings, introductions, meta-questions about the AI
    
    Answer only with the path name.
  `);

    const result = await routerPrompt.pipe(llm).invoke({
        question,
        persona: JSON.stringify(persona)
    });
    const route = (result.content as string).toLowerCase().trim();

    console.log(`- Route: ${route}`);
    return { question, steps: ["route_query"], documents: [] as Document[], route, userPersona: persona };
}

async function hybridRetrieve(state: typeof GraphState.State) {
    console.log("--- HYBRID RETRIEVAL (SEMANTIC + KEYWORD) ---");

    // Multi-query expansion
    const expansionPrompt = ChatPromptTemplate.fromTemplate(`
    Generate 3 search queries for: "{question}"
    Output 3 lines only.
  `);

    const result = await expansionPrompt.pipe(llm).invoke({ question: state.question });
    const queries = (result.content as string).split("\n").filter(q => q.trim().length > 0).slice(0, 3);
    queries.push(state.question);

    const allDocs: Document[] = [];

    // IMPROVEMENT 1: Hybrid Search
    for (const q of queries) {
        // Get embedding
        const embedding = await embeddings.embedQuery(q);

        // Call hybrid search function
        const { data, error } = await supabase.rpc('hybrid_search', {
            query_text: q,
            query_embedding: embedding,
            match_count: 5,
            full_text_weight: 0.3,
            vector_weight: 0.7
        });

        if (!error && data) {
            const docs = data.map((row: any) => new Document({
                pageContent: row.content,
                metadata: { ...row.metadata, similarity: row.similarity }
            }));
            allDocs.push(...docs);
        }
    }

    const uniqueDocs = Array.from(new Map(allDocs.map(doc => [doc.pageContent, doc])).values());
    console.log(`- Retrieved ${uniqueDocs.length} candidates via hybrid search.`);

    return { documents: uniqueDocs, steps: ["hybrid_retrieve"] };
}

async function webSearchParallel(state: typeof GraphState.State) {
    console.log("--- WEB SEARCH BRIDGE ---");
    const searchDoc = new Document({
        pageContent: "General Knowledge: Bitcoin is a decentralized cryptocurrency using blockchain technology and proof-of-work consensus.",
        metadata: { source: "General Knowledge" }
    });
    return { documents: [searchDoc], steps: ["web_search"] };
}

// IMPROVEMENT 4: Contextual Compression
async function compressContext(state: typeof GraphState.State) {
    console.log("--- COMPRESSING CONTEXT ---");
    if (state.documents.length === 0) {
        return { compressedContext: "", steps: ["compress"] };
    }

    const compressionPrompt = ChatPromptTemplate.fromTemplate(`
    Extract ONLY the sentences directly relevant to: "{question}"
    
    Documents:
    {documents}
    
    Output only the relevant excerpts, separated by newlines.
  `);

    const docsText = state.documents.map((d, i) => `[${i}] ${d.pageContent}`).join("\n\n");
    const result = await compressionPrompt.pipe(llm).invoke({
        question: state.question,
        documents: docsText
    });

    const compressed = result.content as string;
    console.log(`- Compressed from ${docsText.length} to ${compressed.length} chars.`);

    return { compressedContext: compressed, steps: ["compress"] };
}

// IMPROVEMENT 3: Citation-Aware Generation
async function generateWithCitations(state: typeof GraphState.State) {
    console.log("--- GENERATING WITH CITATIONS ---");

    const prompt = ChatPromptTemplate.fromTemplate(`
    You are a friendly technical expert.
    
    [INSTRUCTIONS]
    - Be warm and conversational for greetings
    - For technical answers, weave facts naturally WITHOUT saying "Based on the context"
    - When using specific facts from the context, add a citation like [1] after the fact
    - NEVER mention "snippets" or "context" or your technical limitations
    
    User Profile: {persona}
    
    [Chat History]:
    {history}
    
    [Context]: 
    {context}
    
    Question: {question}
    
    Direct Response:
  `);

    const context = state.compressedContext || state.documents.map(d => d.pageContent).join("\n\n");

    const history = state.messages.map(m => {
        const type = (m as any).type || (m as any)._type || (m as any).role || "user";
        return `${type}: ${extractContent(m.content)}`;
    }).join("\n");

    const result = await prompt.pipe(llm).invoke({
        context,
        history,
        question: state.question,
        persona: JSON.stringify(state.userPersona)
    });
    const generation = result.content as string;

    // Extract citations
    const citationMatches = generation.match(/\[\d+\]/g) || [];
    const citations = state.documents.slice(0, 5).map((d, i) =>
        `[${i + 1}] ${d.metadata?.source || "PDF"}: ${d.pageContent.substring(0, 100)}...`
    );

    console.log(`Generated with ${citationMatches.length} citations.`);

    // Update user persona if they introduced themselves
    if (state.question.toLowerCase().includes("my name is")) {
        const nameMatch = state.question.match(/my name is (\w+)/i);
        if (nameMatch) {
            await updateUserPersona("default", {
                ...state.userPersona,
                name: nameMatch[1]
            });
        }
    }

    return {
        generation,
        citations,
        messages: [new AIMessage(generation)],
        steps: ["generate"]
    };
}

async function checkHallucination(state: typeof GraphState.State) {
    console.log("--- HALLUCINATION CHECK ---");
    if (state.route === "conversational") return { hallucinationCheck: "grounded", steps: ["check_hallucination"] };

    const hallucinationPrompt = ChatPromptTemplate.fromTemplate(`
    Does the answer contain FABRICATED facts (specific numbers, names, dates) NOT in the context?
    Paraphrasing and natural language variation is OK.
    
    Context: {context}
    Answer: {generation}
    
    Contains fabrications? (yes/no):
  `);

    const context = state.compressedContext || state.documents.map(d => d.pageContent).join("\n");
    const result = await hallucinationPrompt.pipe(llm).invoke({ context, generation: state.generation });

    const score = (result.content as string).toLowerCase().trim();
    const isFabricated = score.includes("yes");
    console.log(`- Fabricated? ${isFabricated}`);
    return { hallucinationCheck: isFabricated ? "hallucination" : "grounded", steps: ["check_hallucination"] };
}

async function checkAnswer(state: typeof GraphState.State) {
    console.log("--- ANSWER CHECK ---");
    if (state.route === "conversational") return { answerCheck: "useful", steps: ["check_answer"] };

    const answerPrompt = ChatPromptTemplate.fromTemplate(`
    Is this a reasonable response?
    Q: {question} 
    A: {generation}
    Acceptable? (yes/no):
  `);

    const result = await answerPrompt.pipe(llm).invoke({ question: state.question, generation: state.generation });
    const score = (result.content as string).toLowerCase().trim();
    console.log(`- Useful? ${score.includes("yes")}`);
    return { answerCheck: score.includes("yes") ? "useful" : "not_useful", steps: ["check_answer"] };
}

async function transformQuery(state: typeof GraphState.State) {
    console.log("--- QUERY TRANSFORMATION ---");
    const prompt = ChatPromptTemplate.fromTemplate(`Rewrite for better search: {question}`);
    const result = await prompt.pipe(llm).invoke({ question: state.question });
    return { question: result.content as string, steps: ["transform_query"], loopCount: state.loopCount + 1 };
}

// 5. Build Graph
const workflow = new StateGraph(GraphState)
    .addNode("route_query", routeQuery)
    .addNode("hybrid_retrieve", hybridRetrieve)
    .addNode("web_search", webSearchParallel)
    .addNode("compress", compressContext)
    .addNode("generate", generateWithCitations)
    .addNode("check_hallucination", checkHallucination)
    .addNode("check_answer", checkAnswer)
    .addNode("transform_query", transformQuery);

workflow.addEdge(START, "route_query");

workflow.addConditionalEdges("route_query", (state) => {
    return state.route === "conversational" ? "conversational" : "research";
}, {
    conversational: "generate",
    research: "hybrid_retrieve"
});

workflow.addEdge("hybrid_retrieve", "web_search");
workflow.addEdge("web_search", "compress");
workflow.addEdge("compress", "generate");
workflow.addEdge("generate", "check_hallucination");

workflow.addConditionalEdges("check_hallucination", (state) => {
    if (state.route === "conversational") return "finish";
    if (state.hallucinationCheck === "hallucination") return "regenerate";
    return "useful_check";
}, {
    finish: END,
    regenerate: "generate",
    useful_check: "check_answer"
});

workflow.addConditionalEdges("check_answer", (state) => {
    if (state.route === "conversational") return "finish";
    if (state.answerCheck === "not_useful" && state.loopCount < 2) return "retry";
    return "finish";
}, {
    retry: "transform_query",
    finish: END
});

workflow.addEdge("transform_query", "hybrid_retrieve");

await checkpointer.setup();
export const graph = workflow.compile({ checkpointer });

// 6. CLI Support
async function main() {
    const input = process.argv[2] || "What is proof-of-work?";
    const config = { configurable: { thread_id: "advanced-rag-test" } };
    const result = await graph.invoke({ messages: [new HumanMessage(input)] }, config);

    console.log("\n--- Answer ---");
    console.log(result.generation);

    if (result.citations && result.citations.length > 0) {
        console.log("\n--- Sources ---");
        result.citations.forEach((c: string) => console.log(c));
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}
