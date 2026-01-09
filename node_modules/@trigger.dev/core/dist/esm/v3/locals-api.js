// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
import { LocalsAPI } from "./locals/index.js";
/** Entrypoint for runtime API */
export const localsAPI = LocalsAPI.getInstance();
export const locals = {
    create(id) {
        return localsAPI.createLocal(id);
    },
    get(key) {
        return localsAPI.getLocal(key);
    },
    getOrThrow(key) {
        const value = localsAPI.getLocal(key);
        if (!value) {
            throw new Error(`Local with id ${key.id} not found`);
        }
        return value;
    },
    set(key, value) {
        localsAPI.setLocal(key, value);
        return value;
    },
};
//# sourceMappingURL=locals-api.js.map