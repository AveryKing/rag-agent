import { defineConfig } from "@trigger.dev/sdk/v3";

export default defineConfig({
    project: "proj_gsodbjyoytbunyagnezk", // Replace with your actual project ref if needed
    runtime: "node",
    logLevel: "log",
    maxDuration: 60,
    // The directory where your tasks are located
    dirs: ["./src/trigger"],
    build: {
        external: ["pdf-parse", "pdfjs-dist"],
    },
    retries: {
        enabledInDev: true,
        default: {
            maxAttempts: 3,
            minTimeoutInMs: 1000,
            maxTimeoutInMs: 10000,
            factor: 2,
            randomize: true,
        },
    },
});
