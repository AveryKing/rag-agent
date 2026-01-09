"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StandardWaitUntilManager = void 0;
class StandardWaitUntilManager {
    timeoutInMs;
    maybeDeferredPromises = new Set();
    constructor(timeoutInMs = 60_000) {
        this.timeoutInMs = timeoutInMs;
    }
    reset() {
        this.maybeDeferredPromises.clear();
    }
    register(promise) {
        this.maybeDeferredPromises.add(promise);
    }
    async blockUntilSettled() {
        if (this.promisesRequringResolving.length === 0) {
            return;
        }
        const promises = this.promisesRequringResolving.map((p) => typeof p.promise === "function" ? p.promise(this.timeoutInMs) : p.promise);
        await Promise.race([
            Promise.allSettled(promises),
            new Promise((resolve, _) => setTimeout(() => resolve(), this.timeoutInMs)),
        ]);
        this.maybeDeferredPromises.clear();
    }
    requiresResolving() {
        return this.promisesRequringResolving.length > 0;
    }
    get promisesRequringResolving() {
        return Array.from(this.maybeDeferredPromises).filter((p) => p.requiresResolving());
    }
}
exports.StandardWaitUntilManager = StandardWaitUntilManager;
//# sourceMappingURL=manager.js.map