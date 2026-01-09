"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntervalService = void 0;
class IntervalService {
    _onInterval;
    _onError;
    _intervalMs;
    _nextInterval;
    _leadingEdge;
    _isEnabled;
    _isExecuting;
    constructor(opts) {
        this._onInterval = opts.onInterval;
        this._onError = opts.onError;
        this._intervalMs = opts.intervalMs ?? 45_000;
        this._nextInterval = undefined;
        this._leadingEdge = opts.leadingEdge ?? false;
        this._isEnabled = false;
        this._isExecuting = false;
    }
    start() {
        if (this._isEnabled) {
            return;
        }
        this._isEnabled = true;
        if (this._leadingEdge) {
            this.#doInterval();
        }
        else {
            this.#scheduleNextInterval();
        }
    }
    stop() {
        const returnValue = {
            isExecuting: this._isExecuting,
        };
        if (!this._isEnabled) {
            return returnValue;
        }
        this._isEnabled = false;
        this.#clearNextInterval();
        return returnValue;
    }
    resetCurrentInterval() {
        if (!this._isEnabled) {
            return;
        }
        if (this._isExecuting) {
            return;
        }
        this.#clearNextInterval();
        this.#scheduleNextInterval();
    }
    updateInterval(intervalMs) {
        this._intervalMs = intervalMs;
        this.resetCurrentInterval();
    }
    get intervalMs() {
        return this._intervalMs;
    }
    #doInterval = async () => {
        this.#clearNextInterval();
        if (!this._isEnabled) {
            return;
        }
        if (this._isExecuting) {
            console.error("Interval handler already running, skipping");
            return;
        }
        this._isExecuting = true;
        try {
            await this._onInterval();
        }
        catch (error) {
            if (this._onError) {
                try {
                    await this._onError(error);
                }
                catch (error) {
                    console.error("Error during interval error handler", error);
                }
            }
        }
        finally {
            this.#scheduleNextInterval();
            this._isExecuting = false;
        }
    };
    #clearNextInterval() {
        if (this._nextInterval) {
            clearTimeout(this._nextInterval);
        }
    }
    #scheduleNextInterval() {
        this._nextInterval = setTimeout(this.#doInterval, this._intervalMs);
    }
}
exports.IntervalService = IntervalService;
//# sourceMappingURL=interval.js.map