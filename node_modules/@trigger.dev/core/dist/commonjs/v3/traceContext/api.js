"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TraceContextAPI = void 0;
const api_1 = require("@opentelemetry/api");
const globals_js_1 = require("../utils/globals.js");
const API_NAME = "trace-context";
class NoopTraceContextManager {
    getTraceContext() {
        return {};
    }
    reset() { }
    getExternalTraceContext() {
        return undefined;
    }
    extractContext() {
        return api_1.context.active();
    }
    withExternalTrace(fn) {
        return fn();
    }
}
const NOOP_TRACE_CONTEXT_MANAGER = new NoopTraceContextManager();
class TraceContextAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new TraceContextAPI();
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
    getTraceContext() {
        return this.#getManager().getTraceContext();
    }
    getExternalTraceContext() {
        return this.#getManager().getExternalTraceContext();
    }
    extractContext() {
        return this.#getManager().extractContext();
    }
    withExternalTrace(fn) {
        return this.#getManager().withExternalTrace(fn);
    }
    #getManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_TRACE_CONTEXT_MANAGER;
    }
}
exports.TraceContextAPI = TraceContextAPI;
//# sourceMappingURL=api.js.map