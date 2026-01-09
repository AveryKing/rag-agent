import { lifecycleHooks, } from "@trigger.dev/core/v3";
export function onStart(fnOrName, fn) {
    lifecycleHooks.registerGlobalStartHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onStartAttempt(fnOrName, fn) {
    lifecycleHooks.registerGlobalStartAttemptHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onFailure(fnOrName, fn) {
    lifecycleHooks.registerGlobalFailureHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onSuccess(fnOrName, fn) {
    lifecycleHooks.registerGlobalSuccessHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onComplete(fnOrName, fn) {
    lifecycleHooks.registerGlobalCompleteHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onWait(fnOrName, fn) {
    lifecycleHooks.registerGlobalWaitHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onResume(fnOrName, fn) {
    lifecycleHooks.registerGlobalResumeHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
/** @deprecated Use onCatchError instead */
export function onHandleError(fnOrName, fn) {
    onCatchError(fnOrName, fn);
}
export function onCatchError(fnOrName, fn) {
    lifecycleHooks.registerGlobalCatchErrorHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function middleware(fnOrName, fn) {
    lifecycleHooks.registerGlobalMiddlewareHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
export function onCancel(fnOrName, fn) {
    lifecycleHooks.registerGlobalCancelHook({
        id: typeof fnOrName === "string" ? fnOrName : fnOrName.name ? fnOrName.name : undefined,
        fn: typeof fnOrName === "function" ? fnOrName : fn,
    });
}
//# sourceMappingURL=hooks.js.map