import { task } from "@trigger.dev/sdk/v3";
import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRequire } from "module";
const req = createRequire(import.meta.url);

// @ts-ignore
globalThis.__require = { ensure: (_: any, cb: any) => cb(req) };
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OllamaEmbeddings } from "@langchain/ollama";
import { createClient } from "@supabase/supabase-js";
import { SUPABASE_URL, SUPABASE_PRIVATE_KEY, OLLAMA_BASE_URL, EMBEDDING_MODEL } from "../../config.ts";

export const ingestPDFTask = task({
    id: "ingest-pdf",
    run: async (payload: { pdfUrl: string }) => {
        console.log(`Starting ingestion for: ${payload.pdfUrl}`);

        // 1. Initialize Supabase Client
        const supabase = createClient(SUPABASE_URL, SUPABASE_PRIVATE_KEY);

        // 2. Initialize Embeddings
        const embeddings = new OllamaEmbeddings({
            baseUrl: OLLAMA_BASE_URL,
            model: EMBEDDING_MODEL,
        });

        // 3. Load PDF manually using high-level require to bypass bundling errors
        let buffer;
        if (payload.pdfUrl.startsWith("http")) {
            const response = await fetch(payload.pdfUrl);
            buffer = Buffer.from(await response.arrayBuffer());
        } else {
            const fs = await import("fs/promises");
            buffer = await fs.readFile(payload.pdfUrl);
        }

        const pdfParser = req("pdf-parse");
        const data = await pdfParser(buffer);

        const docs = [{
            pageContent: data.text,
            metadata: { pdfUrl: payload.pdfUrl }
        }];
        console.log(`Loaded PDF with ${data.numpages} pages.`);

        // 4. Split text
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const chunks = await splitter.splitDocuments(docs);
        console.log(`Split into ${chunks.length} chunks.`);

        // 5. Store in Supabase
        console.log("Storing in Supabase...");
        await SupabaseVectorStore.fromDocuments(chunks, embeddings, {
            client: supabase,
            tableName: "documents",
            queryName: "match_documents",
        });

        return {
            success: true,
            chunksCount: chunks.length,
        };
    },
});
