"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LocalsAPI = void 0;
const API_NAME = "locals";
const globals_js_1 = require("../utils/globals.js");
const manager_js_1 = require("./manager.js");
const NOOP_LOCALS_MANAGER = new manager_js_1.NoopLocalsManager();
class LocalsAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new LocalsAPI();
        }
        return this._instance;
    }
    setGlobalLocalsManager(localsManager) {
        return (0, globals_js_1.registerGlobal)(API_NAME, localsManager);
    }
    disable() {
        (0, globals_js_1.unregisterGlobal)(API_NAME);
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
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_LOCALS_MANAGER;
    }
}
exports.LocalsAPI = LocalsAPI;
//# sourceMappingURL=index.js.map