const API_NAME = "locals";
import { getGlobal, registerGlobal, unregisterGlobal } from "../utils/globals.js";
import { NoopLocalsManager } from "./manager.js";
const NOOP_LOCALS_MANAGER = new NoopLocalsManager();
export class LocalsAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new LocalsAPI();
        }
        return this._instance;
    }
    setGlobalLocalsManager(localsManager) {
        return registerGlobal(API_NAME, localsManager);
    }
    disable() {
        unregisterGlobal(API_NAME);
    }
    createLocal(id) {
        return this.#getManager().createLocal(id);
    }
    getLocal(key) {
        return this.#getManager().getLocal(key);
    }
    setLocal(key, value) {
        return this.#getManager().setLocal(key, value);
    }
    #getManager() {
        return getGlobal(API_NAME) ?? NOOP_LOCALS_MANAGER;
    }
}
//# sourceMappingURL=index.js.map