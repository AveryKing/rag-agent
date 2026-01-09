import { context } from "@opentelemetry/api";
import { getGlobal, registerGlobal, unregisterGlobal } from "../utils/globals.js";
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
        return context.active();
    }
    withExternalTrace(fn) {
        return fn();
    }
}
const NOOP_TRACE_CONTEXT_MANAGER = new NoopTraceContextManager();
export class TraceContextAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new TraceContextAPI();
        }
        return this._instance;
    }
    setGlobalManager(manager) {
        return registerGlobal(API_NAME, manager);
    }
    disable() {
        unregisterGlobal(API_NAME);
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
        return getGlobal(API_NAME) ?? NOOP_TRACE_CONTEXT_MANAGER;
    }
}
//# sourceMappingURL=api.js.map