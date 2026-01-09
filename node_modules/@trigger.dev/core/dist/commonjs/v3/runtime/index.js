"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RuntimeAPI = void 0;
const API_NAME = "runtime";
const globals_js_1 = require("../utils/globals.js");
const noopRuntimeManager_js_1 = require("./noopRuntimeManager.js");
const usage_api_js_1 = require("../usage-api.js");
const NOOP_RUNTIME_MANAGER = new noopRuntimeManager_js_1.NoopRuntimeManager();
/**
 * All state must be inside the RuntimeManager, do NOT store it on this class.
 * This is because of the "dual package hazard", this can be bundled multiple times.
 */
class RuntimeAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new RuntimeAPI();
        }
        return this._instance;
    }
    waitUntil(waitpointFriendlyId, finishDate) {
        return usage_api_js_1.usage.pauseAsync(() => this.#getRuntimeManager().waitForWaitpoint({ waitpointFriendlyId, finishDate }));
    }
    waitForTask(params) {
        return usage_api_js_1.usage.pauseAsync(() => this.#getRuntimeManager().waitForTask(params));
    }
    waitForToken(waitpointFriendlyId) {
        return usage_api_js_1.usage.pauseAsync(() => this.#getRuntimeManager().waitForWaitpoint({ waitpointFriendlyId }));
    }
    waitForBatch(params) {
        return usage_api_js_1.usage.pauseAsync(() => this.#getRuntimeManager().waitForBatch(params));
    }
    setGlobalRuntimeManager(runtimeManager) {
        return (0, globals_js_1.registerGlobal)(API_NAME, runtimeManager);
    }
    disable() {
        this.#getRuntimeManager().disable();
        (0, globals_js_1.unregisterGlobal)(API_NAME);
    }
    #getRuntimeManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_RUNTIME_MANAGER;
    }
}
exports.RuntimeAPI = RuntimeAPI;
//# sourceMappingURL=index.js.map