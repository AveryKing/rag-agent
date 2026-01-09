import { setInterval } from "node:timers/promises";
import { UsageClient } from "./usageClient.js";
export class ProdUsageManager {
    delegageUsageManager;
    options;
    _measurement;
    _abortController;
    _lastSample;
    _usageClient;
    _initialState = {
        cpuTime: 0,
        costInCents: 0,
    };
    constructor(delegageUsageManager, options) {
        this.delegageUsageManager = delegageUsageManager;
        this.options = options;
        if (this.options.url && this.options.jwt) {
            this._usageClient = new UsageClient(this.options.url, this.options.jwt);
        }
    }
    get isReportingEnabled() {
        return typeof this._usageClient !== "undefined";
    }
    setInitialState(state) {
        this._initialState = state;
    }
    getInitialState() {
        return this._initialState;
    }
    reset() {
        this.delegageUsageManager.reset();
        this._abortController?.abort();
        this._abortController = new AbortController();
        this._usageClient = undefined;
        this._measurement = undefined;
        this._lastSample = undefined;
        this._initialState = {
            cpuTime: 0,
            costInCents: 0,
        };
    }
    disable() {
        this.delegageUsageManager.disable();
        this._abortController?.abort();
    }
    sample() {
        return this._measurement?.sample();
    }
    start() {
        if (!this.isReportingEnabled || !this.options.heartbeatIntervalMs) {
            return this.delegageUsageManager.start();
        }
        if (!this._measurement) {
            this._measurement = this.delegageUsageManager.start();
            this.#startReportingHeartbeat().catch(console.error);
            return this._measurement;
        }
        return this.delegageUsageManager.start();
    }
    stop(measurement) {
        return this.delegageUsageManager.stop(measurement);
    }
    async pauseAsync(cb) {
        return this.delegageUsageManager.pauseAsync(cb);
    }
    async #startReportingHeartbeat() {
        if (!this._measurement || !this.isReportingEnabled || !this.options.heartbeatIntervalMs) {
            return;
        }
        this._abortController = new AbortController();
        try {
            for await (const _ of setInterval(this.options.heartbeatIntervalMs, undefined, {
                signal: this._abortController.signal,
            })) {
                await this.#reportUsage();
            }
        }
        catch (error) {
            if (error instanceof Error && error.name === "AbortError") {
                return;
            }
            throw error;
        }
    }
    async flush() {
        return await this.#reportUsage();
    }
    async #reportUsage() {
        if (!this._measurement) {
            return;
        }
        if (!this.isReportingEnabled) {
            return;
        }
        const client = this._usageClient;
        if (!client) {
            return;
        }
        const sample = this._measurement.sample();
        const cpuTimeSinceLastSample = this._lastSample
            ? sample.cpuTime - this._lastSample.cpuTime
            : sample.cpuTime;
        this._lastSample = sample;
        if (cpuTimeSinceLastSample <= 0) {
            return;
        }
        await client.sendUsageEvent({ durationMs: cpuTimeSinceLastSample });
    }
}
//# sourceMappingURL=prodUsageManager.js.map