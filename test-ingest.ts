import { ingestPDFTask } from "./src/trigger/ingest.ts";

async function test() {
    const pdfUrl = process.argv[2] ?? "https://bitcoin.org/bitcoin.pdf";
    console.log(`Triggering ingestion for: ${pdfUrl}`);

    const handle = await ingestPDFTask.trigger({ pdfUrl });

    console.log("Job triggered successfully!");
    console.log(`Job ID: ${handle.id}`);
    console.log(`Monitor your background task at: https://cloud.trigger.dev/`);
}

test().catch(console.error);
