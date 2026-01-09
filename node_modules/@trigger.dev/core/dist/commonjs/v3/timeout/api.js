"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TimeoutAPI = void 0;
const globals_js_1 = require("../utils/globals.js");
const API_NAME = "timeout";
class NoopTimeoutManager {
    abortAfterTimeout(timeoutInSeconds) {
        return new AbortController();
    }
    reset() { }
}
const NOOP_TIMEOUT_MANAGER = new NoopTimeoutManager();
class TimeoutAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new TimeoutAPI();
        }
        return this._instance;
    }
    get signal() {
        return this.#getManager().signal;
    }
    abortAfterTimeout(timeoutInSeconds) {
        return this.#getManager().abortAfterTimeout(timeoutInSeconds);
    }
    setGlobalManager(manager) {
        return (0, globals_js_1.registerGlobal)(API_NAME, manager);
    }
    disable() {
        (0, globals_js_1.unregisterGlobal)(API_NAME);
    }
    reset() {
        this.#getManager().reset();
        this.disable();
    }
    registerListener(listener) {
        const manager = this.#getManager();
        if (manager.registerListener) {
            manager.registerListener(listener);
        }
    }
    #getManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_TIMEOUT_MANAGER;
    }
}
exports.TimeoutAPI = TimeoutAPI;
//# sourceMappingURL=api.js.map