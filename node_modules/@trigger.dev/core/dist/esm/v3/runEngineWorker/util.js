/** Will ignore headers with falsey values */
export function createHeaders(headersInit) {
    const headers = new Headers();
    for (const [key, value] of Object.entries(headersInit)) {
        if (!value) {
            continue;
        }
        headers.set(key, value);
    }
    return Object.fromEntries(headers.entries());
}
//# sourceMappingURL=util.js.map