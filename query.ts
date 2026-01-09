import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { createClient } from "@supabase/supabase-js";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { SUPABASE_URL, SUPABASE_PRIVATE_KEY, OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL } from "./config.ts";

async function main() {
    const query = process.argv[2];
    if (!query) {
        console.error("Please provide a question.");
        process.exit(1);
    }

    // 1. Setup Supabase
    const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY);

    // Fix: Custom embedding implementation to handle potential input type mismatches
    const embeddings = {
        embedQuery: async (text: any) => {
            // Defensive: Extract string if LangChain passes an object
            const queryText = typeof text === "string" ? text : (text.input || text.pageContent || JSON.stringify(text));

            console.log(`Embedding query: "${queryText.substring(0, 50)}..."`);
            const res = await fetch(`${OLLAMA_BASE_URL}/api/embed`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: EMBEDDING_MODEL, input: [queryText] }),
            });
            if (!res.ok) {
                const err = await res.text();
                console.error("Ollama query embedding Error:", err);
                throw new Error(`Ollama error: ${err}`);
            }
            const data = await res.json();
            return data.embeddings[0];
        },
        embedDocuments: async (texts: string[]) => {
            console.log(`Embedding ${texts.length} documents...`);
            const res = await fetch(`${OLLAMA_BASE_URL}/api/embed`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: EMBEDDING_MODEL, input: texts }),
            });
            if (!res.ok) {
                const err = await res.text();
                console.error("Ollama docs embedding Error:", err);
                throw new Error(`Ollama error: ${err}`);
            }
            const data = await res.json();
            return data.embeddings;
        }
    } as any;

    // 2. Initialize Vector Store
    const vectorStore = new SupabaseVectorStore(embeddings, {
        client: supabase,
        tableName: "documents",
        queryName: "match_documents",
    });

    // 3. Setup LLM
    const llm = new ChatOllama({
        baseUrl: OLLAMA_BASE_URL,
        model: LLM_MODEL,
    });

    // 4. Create Retrieval Chain
    const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's question based only on the provided context. 
    If you don't know the answer, say that you don't know. 
    Don't make up an answer.

    Context: {context}

    Question: {input}
  `);

    const combineDocsChain = await createStuffDocumentsChain({
        llm,
        prompt,
    });

    const retrievalChain = await createRetrievalChain({
        retriever: vectorStore.asRetriever(),
        combineDocsChain,
    });

    // 5. Query
    console.log(`Querying: "${query}"...`);
    const response = await retrievalChain.invoke({
        input: query,
    });

    console.log("\n--- Answer ---");
    console.log(response.answer);

    if (response.context && response.context.length > 0) {
        console.log("\n--- Citations ---");
        response.context.forEach((doc: any, i: number) => {
            console.log(`[${i + 1}] Page ${doc.metadata.loc?.pageNumber || "unknown"}`);
        });
    }
}

main().catch(console.error);
