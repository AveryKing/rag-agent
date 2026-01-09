"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.onStart = onStart;
exports.onStartAttempt = onStartAttempt;
exports.onFailure = onFailure;
exports.onSuccess = onSuccess;
exports.onComplete = onComplete;
exports.onWait = onWait;
exports.onResume = onResume;
exports.onHandleError = onHandleError;
exports.onCatchError = onCatchError;
exports.middleware = middleware;
exports.onCancel = onCancel;
const v3_1 = require("@trigger.dev/core/v3");
function onStart(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalStartHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onStartAttempt(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalStartAttemptHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onFailure(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalFailureHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onSuccess(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalSuccessHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onComplete(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalCompleteHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onWait(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalWaitHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onResume(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalResumeHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
/** @deprecated Use onCatchError instead */
function onHandleError(fnOrName, fn) {
    onCatchError(fnOrName, fn);
}
function onCatchError(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalCatchErrorHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function middleware(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalMiddlewareHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
function onCancel(fnOrName, fn) {
    v3_1.lifecycleHooks.registerGlobalCancelHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
//# sourceMappingURL=hooks.js.map