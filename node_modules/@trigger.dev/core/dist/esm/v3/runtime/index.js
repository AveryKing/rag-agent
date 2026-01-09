const API_NAME = "runtime";
import { getGlobal, registerGlobal, unregisterGlobal } from "../utils/globals.js";
import { NoopRuntimeManager } from "./noopRuntimeManager.js";
import { usage } from "../usage-api.js";
const NOOP_RUNTIME_MANAGER = new NoopRuntimeManager();
/**
 * All state must be inside the RuntimeManager, do NOT store it on this class.
 * This is because of the "dual package hazard", this can be bundled multiple times.
 */
export class RuntimeAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new RuntimeAPI();
        }
        return this._instance;
    }
    waitUntil(waitpointFriendlyId, finishDate) {
        return usage.pauseAsync(() => this.#getRuntimeManager().waitForWaitpoint({ waitpointFriendlyId, finishDate }));
    }
    waitForTask(params) {
        return usage.pauseAsync(() => this.#getRuntimeManager().waitForTask(params));
    }
    waitForToken(waitpointFriendlyId) {
        return usage.pauseAsync(() => this.#getRuntimeManager().waitForWaitpoint({ waitpointFriendlyId }));
    }
    waitForBatch(params) {
        return usage.pauseAsync(() => this.#getRuntimeManager().waitForBatch(params));
    }
    setGlobalRuntimeManager(runtimeManager) {
        return registerGlobal(API_NAME, runtimeManager);
    }
    disable() {
        this.#getRuntimeManager().disable();
        unregisterGlobal(API_NAME);
    }
    #getRuntimeManager() {
        return getGlobal(API_NAME) ?? NOOP_RUNTIME_MANAGER;
    }
}
//# sourceMappingURL=index.js.map