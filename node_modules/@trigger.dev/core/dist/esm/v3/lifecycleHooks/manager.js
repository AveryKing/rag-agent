export class StandardLifecycleHooksManager {
    globalInitHooks = new Map();
    taskInitHooks = new Map();
    globalStartHooks = new Map();
    taskStartHooks = new Map();
    globalStartAttemptHooks = new Map();
    taskStartAttemptHooks = new Map();
    globalFailureHooks = new Map();
    taskFailureHooks = new Map();
    globalSuccessHooks = new Map();
    taskSuccessHooks = new Map();
    globalCompleteHooks = new Map();
    taskCompleteHooks = new Map();
    globalResumeHooks = new Map();
    taskResumeHooks = new Map();
    globalCatchErrorHooks = new Map();
    taskCatchErrorHooks = new Map();
    globalMiddlewareHooks = new Map();
    taskMiddlewareHooks = new Map();
    globalCleanupHooks = new Map();
    taskCleanupHooks = new Map();
    globalWaitHooks = new Map();
    taskWaitHooks = new Map();
    onWaitHookListeners = [];
    onResumeHookListeners = [];
    globalCancelHooks = new Map();
    taskCancelHooks = new Map();
    onCancelHookListeners = [];
    reset() {
        this.onCancelHookListeners.length = 0;
        this.onWaitHookListeners.length = 0;
        this.onResumeHookListeners.length = 0;
    }
    registerOnCancelHookListener(listener) {
        this.onCancelHookListeners.push(listener);
    }
    async callOnCancelHookListeners() {
        await Promise.allSettled(this.onCancelHookListeners.map((listener) => listener()));
    }
    registerOnWaitHookListener(listener) {
        this.onWaitHookListeners.push(listener);
    }
    async callOnWaitHookListeners(wait) {
        await Promise.allSettled(this.onWaitHookListeners.map((listener) => listener(wait)));
    }
    registerOnResumeHookListener(listener) {
        this.onResumeHookListeners.push(listener);
    }
    async callOnResumeHookListeners(wait) {
        await Promise.allSettled(this.onResumeHookListeners.map((listener) => listener(wait)));
    }
    registerGlobalStartHook(hook) {
        const id = generateHookId(hook);
        this.globalStartHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskStartHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskStartHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskStartHook(taskId) {
        return this.taskStartHooks.get(taskId)?.fn;
    }
    getGlobalStartHooks() {
        return Array.from(this.globalStartHooks.values());
    }
    registerGlobalStartAttemptHook(hook) {
        const id = generateHookId(hook);
        this.globalStartAttemptHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskStartAttemptHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskStartAttemptHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskStartAttemptHook(taskId) {
        return this.taskStartAttemptHooks.get(taskId)?.fn;
    }
    getGlobalStartAttemptHooks() {
        return Array.from(this.globalStartAttemptHooks.values());
    }
    registerGlobalInitHook(hook) {
        // if there is no id, lets generate one based on the contents of the function
        const id = generateHookId(hook);
        const registeredHook = {
            id,
            name: hook.id,
            fn: hook.fn,
        };
        this.globalInitHooks.set(id, registeredHook);
    }
    registerTaskInitHook(taskId, hook) {
        const registeredHook = {
            id: generateHookId(hook),
            name: taskId,
            fn: hook.fn,
        };
        this.taskInitHooks.set(taskId, registeredHook);
    }
    getTaskInitHook(taskId) {
        return this.taskInitHooks.get(taskId)?.fn;
    }
    getGlobalInitHooks() {
        return Array.from(this.globalInitHooks.values());
    }
    registerGlobalFailureHook(hook) {
        const id = generateHookId(hook);
        this.globalFailureHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskFailureHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskFailureHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskFailureHook(taskId) {
        return this.taskFailureHooks.get(taskId)?.fn;
    }
    getGlobalFailureHooks() {
        return Array.from(this.globalFailureHooks.values());
    }
    registerGlobalSuccessHook(hook) {
        const id = generateHookId(hook);
        this.globalSuccessHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskSuccessHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskSuccessHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskSuccessHook(taskId) {
        return this.taskSuccessHooks.get(taskId)?.fn;
    }
    getGlobalSuccessHooks() {
        return Array.from(this.globalSuccessHooks.values());
    }
    registerGlobalCompleteHook(hook) {
        const id = generateHookId(hook);
        this.globalCompleteHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskCompleteHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskCompleteHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskCompleteHook(taskId) {
        return this.taskCompleteHooks.get(taskId)?.fn;
    }
    getGlobalCompleteHooks() {
        return Array.from(this.globalCompleteHooks.values());
    }
    registerGlobalWaitHook(hook) {
        const id = generateHookId(hook);
        this.globalWaitHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskWaitHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskWaitHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskWaitHook(taskId) {
        return this.taskWaitHooks.get(taskId)?.fn;
    }
    getGlobalWaitHooks() {
        return Array.from(this.globalWaitHooks.values());
    }
    registerGlobalResumeHook(hook) {
        const id = generateHookId(hook);
        this.globalResumeHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskResumeHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskResumeHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskResumeHook(taskId) {
        return this.taskResumeHooks.get(taskId)?.fn;
    }
    getGlobalResumeHooks() {
        return Array.from(this.globalResumeHooks.values());
    }
    registerGlobalCatchErrorHook(hook) {
        const id = generateHookId(hook);
        this.globalCatchErrorHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskCatchErrorHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskCatchErrorHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskCatchErrorHook(taskId) {
        return this.taskCatchErrorHooks.get(taskId)?.fn;
    }
    getGlobalCatchErrorHooks() {
        return Array.from(this.globalCatchErrorHooks.values());
    }
    registerGlobalMiddlewareHook(hook) {
        const id = generateHookId(hook);
        this.globalMiddlewareHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskMiddlewareHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskMiddlewareHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskMiddlewareHook(taskId) {
        return this.taskMiddlewareHooks.get(taskId)?.fn;
    }
    getGlobalMiddlewareHooks() {
        return Array.from(this.globalMiddlewareHooks.values());
    }
    registerGlobalCleanupHook(hook) {
        const id = generateHookId(hook);
        this.globalCleanupHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskCleanupHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskCleanupHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getTaskCleanupHook(taskId) {
        return this.taskCleanupHooks.get(taskId)?.fn;
    }
    getGlobalCleanupHooks() {
        return Array.from(this.globalCleanupHooks.values());
    }
    registerGlobalCancelHook(hook) {
        const id = generateHookId(hook);
        this.globalCancelHooks.set(id, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    registerTaskCancelHook(taskId, hook) {
        const id = generateHookId(hook);
        this.taskCancelHooks.set(taskId, {
            id,
            name: hook.id,
            fn: hook.fn,
        });
    }
    getGlobalCancelHooks() {
        return Array.from(this.globalCancelHooks.values());
    }
    getTaskCancelHook(taskId) {
        return this.taskCancelHooks.get(taskId)?.fn;
    }
}
export class NoopLifecycleHooksManager {
    registerOnCancelHookListener(listener) {
        // Noop
    }
    async callOnCancelHookListeners() {
        // Noop
    }
    registerGlobalCancelHook(hook) { }
    registerTaskCancelHook(taskId, hook) {
        // Noop
    }
    getTaskCancelHook(taskId) {
        return undefined;
    }
    getGlobalCancelHooks() {
        return [];
    }
    registerOnWaitHookListener(listener) {
        // Noop
    }
    async callOnWaitHookListeners(wait) {
        // Noop
    }
    registerOnResumeHookListener(listener) {
        // Noop
    }
    async callOnResumeHookListeners(wait) {
        // Noop
    }
    registerGlobalInitHook(hook) {
        // Noop
    }
    registerTaskInitHook(taskId, hook) {
        // Noop
    }
    getTaskInitHook(taskId) {
        return undefined;
    }
    getGlobalInitHooks() {
        return [];
    }
    registerGlobalStartHook(hook) {
        // Noop
    }
    registerTaskStartHook(taskId, hook) {
        // Noop
    }
    getTaskStartHook(taskId) {
        return undefined;
    }
    getGlobalStartHooks() {
        return [];
    }
    registerGlobalStartAttemptHook() {
        // Noop
    }
    registerTaskStartAttemptHook() {
        // Noop
    }
    getTaskStartAttemptHook() {
        return undefined;
    }
    getGlobalStartAttemptHooks() {
        return [];
    }
    registerGlobalFailureHook(hook) {
        // Noop
    }
    registerTaskFailureHook(taskId, hook) {
        // Noop
    }
    getTaskFailureHook(taskId) {
        return undefined;
    }
    getGlobalFailureHooks() {
        return [];
    }
    registerGlobalSuccessHook(hook) {
        // Noop
    }
    registerTaskSuccessHook(taskId, hook) {
        // Noop
    }
    getTaskSuccessHook(taskId) {
        return undefined;
    }
    getGlobalSuccessHooks() {
        return [];
    }
    registerGlobalCompleteHook(hook) {
        // Noop
    }
    registerTaskCompleteHook(taskId, hook) {
        // Noop
    }
    getTaskCompleteHook(taskId) {
        return undefined;
    }
    getGlobalCompleteHooks() {
        return [];
    }
    registerGlobalWaitHook(hook) {
        // Noop
    }
    registerTaskWaitHook(taskId, hook) {
        // Noop
    }
    getTaskWaitHook(taskId) {
        return undefined;
    }
    getGlobalWaitHooks() {
        return [];
    }
    registerGlobalResumeHook(hook) {
        // Noop
    }
    registerTaskResumeHook(taskId, hook) {
        // Noop
    }
    getTaskResumeHook(taskId) {
        return undefined;
    }
    getGlobalResumeHooks() {
        return [];
    }
    registerGlobalCatchErrorHook() {
        // Noop
    }
    registerTaskCatchErrorHook() {
        // Noop
    }
    getTaskCatchErrorHook() {
        return undefined;
    }
    getGlobalCatchErrorHooks() {
        return [];
    }
    registerGlobalMiddlewareHook() {
        // Noop
    }
    registerTaskMiddlewareHook() {
        // Noop
    }
    getTaskMiddlewareHook() {
        return undefined;
    }
    getGlobalMiddlewareHooks() {
        return [];
    }
    registerGlobalCleanupHook(hook) {
        // Noop
    }
    registerTaskCleanupHook(taskId, hook) {
        // Noop
    }
    getTaskCleanupHook(taskId) {
        return undefined;
    }
    getGlobalCleanupHooks() {
        return [];
    }
}
function generateHookId(hook) {
    return hook.id ?? hook.fn.toString();
}
//# sourceMappingURL=manager.js.map