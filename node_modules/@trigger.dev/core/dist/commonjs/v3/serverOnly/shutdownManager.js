"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.shutdownManager = exports.ShutdownManager = void 0;
const std_env_1 = require("std-env");
const structuredLogger_js_1 = require("../utils/structuredLogger.js");
const singleton_js_1 = require("./singleton.js");
class ShutdownManager {
    disableForTesting;
    isShuttingDown = false;
    signalNumbers = {
        SIGINT: 2,
        SIGTERM: 15,
    };
    logger = new structuredLogger_js_1.SimpleStructuredLogger("shutdownManager");
    handlers = new Map();
    constructor(disableForTesting = true) {
        this.disableForTesting = disableForTesting;
        if (disableForTesting)
            return;
        process.on("SIGTERM", () => this.shutdown("SIGTERM"));
        process.on("SIGINT", () => this.shutdown("SIGINT"));
    }
    register(name, handler, signals = ["SIGTERM", "SIGINT"]) {
        if (!this.isEnabled())
            return;
        if (this.handlers.has(name)) {
            throw new Error(`Shutdown handler "${name}" already registered`);
        }
        this.handlers.set(name, { handler, signals });
    }
    unregister(name) {
        if (!this.isEnabled())
            return;
        if (!this.handlers.has(name)) {
            throw new Error(`Shutdown handler "${name}" not registered`);
        }
        this.handlers.delete(name);
    }
    async shutdown(signal) {
        if (!this.isEnabled())
            return;
        if (this.isShuttingDown)
            return;
        this.isShuttingDown = true;
        this.logger.info(`Received ${signal}. Starting graceful shutdown...`);
        // Get handlers that are registered for this signal
        const handlersToRun = Array.from(this.handlers.entries()).filter(([_, { signals }]) => signals.includes(signal));
        try {
            const results = await Promise.allSettled(handlersToRun.map(async ([name, { handler }]) => {
                try {
                    this.logger.info(`Running shutdown handler: ${name}`);
                    await handler(signal);
                    this.logger.info(`Shutdown handler completed: ${name}`);
                }
                catch (error) {
                    this.logger.error(`Shutdown handler failed: ${name}`, { error });
                    throw error;
                }
            }));
            // Log any failures
            results.forEach((result, index) => {
                if (result.status === "rejected") {
                    const handlerEntry = handlersToRun[index];
                    if (handlerEntry) {
                        const [name] = handlerEntry;
                        this.logger.error(`Shutdown handler "${name}" failed:`, { reason: result.reason });
                    }
                }
            });
        }
        catch (error) {
            this.logger.error("Error during shutdown:", { error });
        }
        finally {
            // Exit with the correct signal number
            process.exit(128 + this.signalNumbers[signal]);
        }
    }
    isEnabled() {
        if (!this.disableForTesting) {
            return true;
        }
        return !std_env_1.isTest;
    }
    // Only for testing
    _getHandlersForTesting() {
        return new Map(this.handlers);
    }
}
exports.ShutdownManager = ShutdownManager;
exports.shutdownManager = (0, singleton_js_1.singleton)("shutdownManager", () => new ShutdownManager());
//# sourceMappingURL=shutdownManager.js.map