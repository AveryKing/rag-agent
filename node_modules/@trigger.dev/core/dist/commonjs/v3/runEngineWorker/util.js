"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createHeaders = createHeaders;
/** Will ignore headers with falsey values */
function createHeaders(headersInit) {
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