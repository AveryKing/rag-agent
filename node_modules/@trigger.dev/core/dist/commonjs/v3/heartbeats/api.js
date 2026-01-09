"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HeartbeatsAPI = void 0;
const globals_js_1 = require("../utils/globals.js");
const API_NAME = "heartbeats";
class NoopHeartbeatsManager {
    startHeartbeat(id) {
        return;
    }
    stopHeartbeat() {
        return;
    }
    async yield() {
        return;
    }
    get lastHeartbeat() {
        return undefined;
    }
    reset() { }
}
const NOOP_HEARTBEATS_MANAGER = new NoopHeartbeatsManager();
class HeartbeatsAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new HeartbeatsAPI();
        }
        return this._instance;
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
    get lastHeartbeat() {
        return this.#getManager().lastHeartbeat;
    }
    startHeartbeat(id) {
        return this.#getManager().startHeartbeat(id);
    }
    stopHeartbeat() {
        return this.#getManager().stopHeartbeat();
    }
    yield() {
        return this.#getManager().yield();
    }
    #getManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_HEARTBEATS_MANAGER;
    }
}
exports.HeartbeatsAPI = HeartbeatsAPI;
//# sourceMappingURL=api.js.map