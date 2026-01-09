import { Annotation, StateGraph, END, START } from "@langchain/langgraph";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { ChatOllama } from "@langchain/ollama";
import { createClient } from "@supabase/supabase-js";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { SUPABASE_URL, SUPABASE_PRIVATE_KEY, OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL } from "./config.ts";

// 1. Definition of Graph State
const GraphState = Annotation.Root({
    question: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    documents: Annotation<Document[]>({
        reducer: (x, y) => y ?? x,
    }),
    generation: Annotation<string>({
        reducer: (x, y) => y ?? x,
    }),
    steps: Annotation<string[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    loopCount: Annotation<number>({
        reducer: (x, y) => y ?? x,
        default: () => 0,
    }),
});

// 2. Setup Clients
const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY);

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
});

// 3. Define Nodes

async function retrieve(state: typeof GraphState.State) {
    console.log("--- RETRIEVING ---");
    const documents = await vectorStore.similaritySearch(state.question, 3);
    return {
        documents,
        steps: ["retrieve"],
        loopCount: state.loopCount + 1
    };
}

async function gradeDocuments(state: typeof GraphState.State) {
    console.log("--- GRADING DOCUMENTS ---");

    const graderPrompt = ChatPromptTemplate.fromTemplate(`
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a perfect answer to be relevant.
    
    User Question: {question}
    Retrieved Document: {document}
    
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
    Answer only with 'yes' or 'no'.
  `);

    const relevantDocs: Document[] = [];
    for (const doc of state.documents) {
        const chain = graderPrompt.pipe(llm);
        const result = await chain.invoke({
            question: state.question,
            document: doc.pageContent,
        });

        const score = (result.content as string).toLowerCase().trim();
        if (score.includes("yes")) {
            relevantDocs.push(doc);
        }
    }

    return { documents: relevantDocs, steps: ["grade_documents"] };
}

async function generate(state: typeof GraphState.State) {
    console.log("--- GENERATING ---");

    if (state.documents.length === 0) {
        return {
            generation: "I'm sorry, but I couldn't find any relevant information in the provided documents to answer that question.",
            steps: ["generate"]
        };
    }

    const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's question based only on the provided context.
    If you don't know the answer, say that you don't know.
    
    Context: {context}
    Question: {question}
  `);

    const context = state.documents.map((d) => d.pageContent).join("\n\n");
    console.log(`Context used for generation (first 100 chars): "${context.substring(0, 100)}..."`);

    const chain = prompt.pipe(llm);
    const result = await chain.invoke({
        context,
        question: state.question,
    });

    return { generation: result.content as string, steps: ["generate"] };
}

async function transformQuery(state: typeof GraphState.State) {
    console.log("--- TRANSFORMING QUERY ---");
    const prompt = ChatPromptTemplate.fromTemplate(`
    You are a query rewriter that optimizes questions for vector search.
    The original question did not yield relevant results.
    Look at the question and try to reason about the underlying intent.
    
    Original Question: {question}
    
    Instruction: Output only the improved question text.
  `);

    const chain = prompt.pipe(llm);
    const result = await chain.invoke({ question: state.question });

    return { question: result.content as string, steps: ["transform_query"] };
}

// 4. Define Edges (Decision Points)

function decideNextStep(state: typeof GraphState.State) {
    if (state.documents.length === 0) {
        if (state.loopCount >= 2) {
            console.log("--- DECISION: MAX LOOPS REACHED ---");
            return "generate";
        }
        console.log("--- DECISION: TRANSFORM QUERY ---");
        return "transform_query";
    }
    console.log("--- DECISION: GENERATE ---");
    return "generate";
}

// 5. Build Graph
const workflow = new StateGraph(GraphState)
    .addNode("retrieve", retrieve)
    .addNode("grade_documents", gradeDocuments)
    .addNode("generate", generate)
    .addNode("transform_query", transformQuery)

    .addEdge(START, "retrieve")
    .addEdge("retrieve", "grade_documents")
    .addConditionalEdges("grade_documents", decideNextStep, {
        transform_query: "transform_query",
        generate: "generate",
    })
    .addEdge("transform_query", "retrieve") // Loop back to search again
    .addEdge("generate", END);

const app = workflow.compile();

// 6. Run Execution
async function main() {
    const inputQuestion = process.argv[2] || "Who is Antigravity?";
    console.log(`\nQuestion: ${inputQuestion}`);

    const result = await app.invoke({ question: inputQuestion });

    console.log("\n--- Final Generation ---");
    console.log(result.generation);
    console.log("\n--- Steps Taken ---");
    console.log(result.steps.join(" -> "));
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}

export { app };
