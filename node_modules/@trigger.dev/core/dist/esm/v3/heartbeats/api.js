import { getGlobal, registerGlobal, unregisterGlobal } from "../utils/globals.js";
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
export class HeartbeatsAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new HeartbeatsAPI();
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
        return getGlobal(API_NAME) ?? NOOP_HEARTBEATS_MANAGER;
    }
}
//# sourceMappingURL=api.js.map