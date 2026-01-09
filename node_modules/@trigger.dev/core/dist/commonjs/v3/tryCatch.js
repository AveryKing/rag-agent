"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.tryCatch = tryCatch;
// Main wrapper function
async function tryCatch(promise) {
    try {
        const data = await promise;
        return [null, data];
    }
    catch (error) {
        return [error, null];
    }
}
//# sourceMappingURL=tryCatch.js.map