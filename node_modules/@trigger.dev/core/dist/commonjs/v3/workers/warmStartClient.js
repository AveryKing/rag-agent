"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WarmStartClient = void 0;
const runEngine_js_1 = require("../schemas/runEngine.js");
const structuredLogger_js_1 = require("../utils/structuredLogger.js");
const warmStart_js_1 = require("../schemas/warmStart.js");
const zodfetch_js_1 = require("../zodfetch.js");
const backoff_js_1 = require("../apps/backoff.js");
class WarmStartClient {
    opts;
    logger = new structuredLogger_js_1.SimpleStructuredLogger("warm-start-client");
    apiUrl;
    backoff = new backoff_js_1.ExponentialBackoff("FullJitter");
    abortController = null;
    get connectUrl() {
        return new URL("/connect", this.apiUrl);
    }
    get warmStartUrl() {
        return new URL("/warm-start", this.apiUrl);
    }
    constructor(opts) {
        this.opts = opts;
        this.apiUrl = opts.apiUrl;
    }
    abort() {
        if (!this.abortController) {
            this.logger.warn("Abort called but no abort controller exists");
            return;
        }
        this.abortController.abort();
        this.abortController = null;
    }
    async withAbort(fn) {
        if (this.abortController) {
            throw new Error("A warm start is already in progress");
        }
        this.abortController = new AbortController();
        try {
            return await fn(this.abortController.signal);
        }
        finally {
            this.abortController = null;
        }
    }
    async connect() {
        return (0, zodfetch_js_1.wrapZodFetch)(warmStart_js_1.WarmStartConnectResponse, this.connectUrl.href, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        }, {
            retry: {
                minTimeoutInMs: 200,
                maxTimeoutInMs: 2000,
                maxAttempts: 3,
                factor: 2,
                randomize: false,
            },
        });
    }
    async warmStart({ workerInstanceName, connectionTimeoutMs, keepaliveMs, }) {
        return this.withAbort(async (abortSignal) => {
            const res = await this.longPoll(this.warmStartUrl.href, {
                method: "GET",
                headers: {
                    "x-trigger-workload-controller-id": this.opts.controllerId,
                    "x-trigger-deployment-id": this.opts.deploymentId,
                    "x-trigger-deployment-version": this.opts.deploymentVersion,
                    "x-trigger-machine-cpu": this.opts.machineCpu,
                    "x-trigger-machine-memory": this.opts.machineMemory,
                    "x-trigger-worker-instance-name": workerInstanceName,
                },
            }, {
                timeoutMs: connectionTimeoutMs,
                totalDurationMs: keepaliveMs,
                abortSignal,
            });
            if (!res.ok) {
                this.logger.error("warmStart: failed", {
                    error: res.error,
                    connectionTimeoutMs,
                    keepaliveMs,
                });
                return null;
            }
            const nextRun = runEngine_js_1.DequeuedMessage.parse(res.data);
            this.logger.debug("warmStart: got next run", { nextRun });
            return nextRun;
        });
    }
    async longPoll(url, requestInit, { timeoutMs, totalDurationMs, abortSignal, }) {
        this.logger.debug("Long polling", { url, requestInit, timeoutMs, totalDurationMs });
        const endTime = Date.now() + totalDurationMs;
        let retries = 0;
        while (Date.now() < endTime) {
            if (abortSignal.aborted) {
                return {
                    ok: false,
                    error: "Aborted - abort signal triggered before fetch",
                };
            }
            try {
                const timeoutController = new AbortController();
                const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
                // Create compound signal that aborts on either timeout or parent abort
                const signals = [timeoutController.signal, abortSignal];
                const signal = AbortSignal.any(signals);
                const response = await fetch(url, { ...requestInit, signal });
                clearTimeout(timeoutId);
                if (response.ok) {
                    const data = await response.json();
                    return {
                        ok: true,
                        data,
                    };
                }
                else {
                    return {
                        ok: false,
                        error: `Server error: ${response.status}`,
                    };
                }
            }
            catch (error) {
                if (error instanceof Error && error.name === "AbortError") {
                    // Check if this was a parent abort or just a timeout
                    if (abortSignal.aborted) {
                        return {
                            ok: false,
                            error: "Aborted - abort signal triggered during fetch",
                        };
                    }
                    this.logger.log("Long poll request timed out, retrying...");
                    continue;
                }
                else {
                    this.logger.error("Error during fetch, retrying...", { error });
                    // Wait with exponential backoff
                    await this.backoff.wait(retries++);
                    continue;
                }
            }
        }
        return {
            ok: false,
            error: "TotalDurationExceeded",
        };
    }
}
exports.WarmStartClient = WarmStartClient;
//# sourceMappingURL=warmStartClient.js.map