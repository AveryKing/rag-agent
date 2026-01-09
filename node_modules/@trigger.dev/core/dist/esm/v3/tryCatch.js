// Main wrapper function
export async function tryCatch(promise) {
    try {
        const data = await promise;
        return [null, data];
    }
    catch (error) {
        return [error, null];
    }
}
//# sourceMappingURL=tryCatch.js.map