"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunMetadataAPI = void 0;
const globals_js_1 = require("../utils/globals.js");
const noopManager_js_1 = require("./noopManager.js");
const API_NAME = "run-metadata";
const NOOP_MANAGER = new noopManager_js_1.NoopRunMetadataManager();
class RunMetadataAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new RunMetadataAPI();
        }
        return this._instance;
    }
    setGlobalManager(manager) {
        return (0, globals_js_1.registerGlobal)(API_NAME, manager);
    }
    #getManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_MANAGER;
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
exports.RunMetadataAPI = RunMetadataAPI;
//# sourceMappingURL=index.js.map