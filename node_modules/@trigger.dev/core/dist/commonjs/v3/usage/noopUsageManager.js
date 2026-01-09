"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NoopUsageManager = void 0;
class NoopUsageManager {
    disable() {
        // Noop
    }
    async flush() {
        // Noop
    }
    start() {
        return {
            sample: () => ({ cpuTime: 0, wallTime: 0 }),
        };
    }
    stop(measurement) {
        return measurement.sample();
    }
    pauseAsync(cb) {
        return cb();
    }
    sample() {
        return undefined;
    }
    reset() {
        // Noop
    }
    getInitialState() {
        return {
            cpuTime: 0,
            costInCents: 0,
        };
    }
}
exports.NoopUsageManager = NoopUsageManager;
//# sourceMappingURL=noopUsageManager.js.map