"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LifecycleHooksAPI = void 0;
const API_NAME = "lifecycle-hooks";
const globals_js_1 = require("../utils/globals.js");
const manager_js_1 = require("./manager.js");
const NOOP_LIFECYCLE_HOOKS_MANAGER = new manager_js_1.NoopLifecycleHooksManager();
class LifecycleHooksAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new LifecycleHooksAPI();
        }
        return this._instance;
    }
    setGlobalLifecycleHooksManager(lifecycleHooksManager) {
        return (0, globals_js_1.registerGlobal)(API_NAME, lifecycleHooksManager);
    }
    disable() {
        (0, globals_js_1.unregisterGlobal)(API_NAME);
    }
    registerGlobalInitHook(hook) {
        this.#getManager().registerGlobalInitHook(hook);
    }
    registerTaskInitHook(taskId, hook) {
        this.#getManager().registerTaskInitHook(taskId, hook);
    }
    getTaskInitHook(taskId) {
        return this.#getManager().getTaskInitHook(taskId);
    }
    getGlobalInitHooks() {
        return this.#getManager().getGlobalInitHooks();
    }
    registerTaskStartHook(taskId, hook) {
        this.#getManager().registerTaskStartHook(taskId, hook);
    }
    registerGlobalStartHook(hook) {
        this.#getManager().registerGlobalStartHook(hook);
    }
    getTaskStartHook(taskId) {
        return this.#getManager().getTaskStartHook(taskId);
    }
    getGlobalStartHooks() {
        return this.#getManager().getGlobalStartHooks();
    }
    registerTaskStartAttemptHook(taskId, hook) {
        this.#getManager().registerTaskStartAttemptHook(taskId, hook);
    }
    registerGlobalStartAttemptHook(hook) {
        this.#getManager().registerGlobalStartAttemptHook(hook);
    }
    getTaskStartAttemptHook(taskId) {
        return this.#getManager().getTaskStartAttemptHook(taskId);
    }
    getGlobalStartAttemptHooks() {
        return this.#getManager().getGlobalStartAttemptHooks();
    }
    registerGlobalFailureHook(hook) {
        this.#getManager().registerGlobalFailureHook(hook);
    }
    registerTaskFailureHook(taskId, hook) {
        this.#getManager().registerTaskFailureHook(taskId, hook);
    }
    getTaskFailureHook(taskId) {
        return this.#getManager().getTaskFailureHook(taskId);
    }
    getGlobalFailureHooks() {
        return this.#getManager().getGlobalFailureHooks();
    }
    registerGlobalSuccessHook(hook) {
        this.#getManager().registerGlobalSuccessHook(hook);
    }
    registerTaskSuccessHook(taskId, hook) {
        this.#getManager().registerTaskSuccessHook(taskId, hook);
    }
    getTaskSuccessHook(taskId) {
        return this.#getManager().getTaskSuccessHook(taskId);
    }
    getGlobalSuccessHooks() {
        return this.#getManager().getGlobalSuccessHooks();
    }
    registerGlobalCompleteHook(hook) {
        this.#getManager().registerGlobalCompleteHook(hook);
    }
    registerTaskCompleteHook(taskId, hook) {
        this.#getManager().registerTaskCompleteHook(taskId, hook);
    }
    getTaskCompleteHook(taskId) {
        return this.#getManager().getTaskCompleteHook(taskId);
    }
    getGlobalCompleteHooks() {
        return this.#getManager().getGlobalCompleteHooks();
    }
    registerGlobalWaitHook(hook) {
        this.#getManager().registerGlobalWaitHook(hook);
    }
    registerTaskWaitHook(taskId, hook) {
        this.#getManager().registerTaskWaitHook(taskId, hook);
    }
    getTaskWaitHook(taskId) {
        return this.#getManager().getTaskWaitHook(taskId);
    }
    getGlobalWaitHooks() {
        return this.#getManager().getGlobalWaitHooks();
    }
    registerGlobalResumeHook(hook) {
        this.#getManager().registerGlobalResumeHook(hook);
    }
    registerTaskResumeHook(taskId, hook) {
        this.#getManager().registerTaskResumeHook(taskId, hook);
    }
    getTaskResumeHook(taskId) {
        return this.#getManager().getTaskResumeHook(taskId);
    }
    getGlobalResumeHooks() {
        return this.#getManager().getGlobalResumeHooks();
    }
    registerGlobalCatchErrorHook(hook) {
        this.#getManager().registerGlobalCatchErrorHook(hook);
    }
    registerTaskCatchErrorHook(taskId, hook) {
        this.#getManager().registerTaskCatchErrorHook(taskId, hook);
    }
    getTaskCatchErrorHook(taskId) {
        return this.#getManager().getTaskCatchErrorHook(taskId);
    }
    getGlobalCatchErrorHooks() {
        return this.#getManager().getGlobalCatchErrorHooks();
    }
    registerGlobalMiddlewareHook(hook) {
        this.#getManager().registerGlobalMiddlewareHook(hook);
    }
    registerTaskMiddlewareHook(taskId, hook) {
        this.#getManager().registerTaskMiddlewareHook(taskId, hook);
    }
    getTaskMiddlewareHook(taskId) {
        return this.#getManager().getTaskMiddlewareHook(taskId);
    }
    getGlobalMiddlewareHooks() {
        return this.#getManager().getGlobalMiddlewareHooks();
    }
    registerGlobalCleanupHook(hook) {
        this.#getManager().registerGlobalCleanupHook(hook);
    }
    registerTaskCleanupHook(taskId, hook) {
        this.#getManager().registerTaskCleanupHook(taskId, hook);
    }
    getTaskCleanupHook(taskId) {
        return this.#getManager().getTaskCleanupHook(taskId);
    }
    getGlobalCleanupHooks() {
        return this.#getManager().getGlobalCleanupHooks();
    }
    callOnWaitHookListeners(wait) {
        return this.#getManager().callOnWaitHookListeners(wait);
    }
    callOnResumeHookListeners(wait) {
        return this.#getManager().callOnResumeHookListeners(wait);
    }
    registerOnWaitHookListener(listener) {
        this.#getManager().registerOnWaitHookListener(listener);
    }
    registerOnResumeHookListener(listener) {
        this.#getManager().registerOnResumeHookListener(listener);
    }
    registerGlobalCancelHook(hook) {
        this.#getManager().registerGlobalCancelHook(hook);
    }
    registerTaskCancelHook(taskId, hook) {
        this.#getManager().registerTaskCancelHook(taskId, hook);
    }
    getTaskCancelHook(taskId) {
        return this.#getManager().getTaskCancelHook(taskId);
    }
    getGlobalCancelHooks() {
        return this.#getManager().getGlobalCancelHooks();
    }
    callOnCancelHookListeners() {
        return this.#getManager().callOnCancelHookListeners();
    }
    registerOnCancelHookListener(listener) {
        this.#getManager().registerOnCancelHookListener(listener);
    }
    #getManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_LIFECYCLE_HOOKS_MANAGER;
    }
}
exports.LifecycleHooksAPI = LifecycleHooksAPI;
//# sourceMappingURL=index.js.map