import { ingestPDFTask } from "./src/trigger/ingest.ts";

async function test() {
    console.log("Triggering test ingestion...");

    // Use the sample.pdf file path or a URL
    // PDFLoader can take a path or a Blob. For a local file, we use path.resolve.
    const pdfUrl = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";

    const handle = await ingestPDFTask.trigger({ pdfUrl });

    console.log("Job triggered!");
    console.log(`Check your progress at: https://cloud.trigger.dev/jobs/${handle.id}`);
}

test().catch(console.error);
