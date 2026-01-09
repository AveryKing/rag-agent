"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.locals = exports.localsAPI = void 0;
// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
const index_js_1 = require("./locals/index.js");
/** Entrypoint for runtime API */
exports.localsAPI = index_js_1.LocalsAPI.getInstance();
exports.locals = {
    create(id) {
        return exports.localsAPI.createLocal(id);
    },
    get(key) {
        return exports.localsAPI.getLocal(key);
    },
    getOrThrow(key) {
        const value = exports.localsAPI.getLocal(key);
        if (!value) {
            throw new Error(`Local with id ${key.id} not found`);
        }
        return value;
    },
    set(key, value) {
        exports.localsAPI.setLocal(key, value);
        return value;
    },
};
//# sourceMappingURL=locals-api.js.map