"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.randomUUID = randomUUID;
exports.digestSHA256 = digestSHA256;
async function randomUUID() {
    const { randomUUID } = await import("uncrypto");
    return randomUUID();
}
async function digestSHA256(data) {
    const { subtle } = await import("uncrypto");
    const hash = await subtle.digest("SHA-256", new TextEncoder().encode(data));
    // Return a hex string, using cross-runtime compatible methods
    return Array.from(new Uint8Array(hash))
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
}
//# sourceMappingURL=crypto.js.map