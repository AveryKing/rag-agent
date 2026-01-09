import { getGlobal, registerGlobal } from "../utils/globals.js";
import { NoopRunMetadataManager } from "./noopManager.js";
const API_NAME = "run-metadata";
const NOOP_MANAGER = new NoopRunMetadataManager();
export class RunMetadataAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new RunMetadataAPI();
        }
        return this._instance;
    }
    setGlobalManager(manager) {
        return registerGlobal(API_NAME, manager);
    }
    #getManager() {
        return getGlobal(API_NAME) ?? NOOP_MANAGER;
    }
    enterWithMetadata(metadata) {
        this.#getManager().enterWithMetadata(metadata);
    }
    current() {
        return this.#getManager().current();
    }
    getKey(key) {
        return this.#getManager().getKey(key);
    }
    set(key, value) {
        this.#getManager().set(key, value);
        return this;
    }
    del(key) {
        this.#getManager().del(key);
        return this;
    }
    increment(key, value) {
        this.#getManager().increment(key, value);
        return this;
    }
    decrement(key, value) {
        this.#getManager().decrement(key, value);
        return this;
    }
    append(key, value) {
        this.#getManager().append(key, value);
        return this;
    }
    remove(key, value) {
        this.#getManager().remove(key, value);
        return this;
    }
    update(metadata) {
        this.#getManager().update(metadata);
        return this;
    }
    stream(key, value, signal) {
        return this.#getManager().stream(key, value, signal);
    }
    fetchStream(key, signal) {
        return this.#getManager().fetchStream(key, signal);
    }
    flush(requestOptions) {
        return this.#getManager().flush(requestOptions);
    }
    refresh(requestOptions) {
        return this.#getManager().refresh(requestOptions);
    }
    get parent() {
        return this.#getManager().parent;
    }
    get root() {
        return this.#getManager().root;
    }
}
//# sourceMappingURL=index.js.map