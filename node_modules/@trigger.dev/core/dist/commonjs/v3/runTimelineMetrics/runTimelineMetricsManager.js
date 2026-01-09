"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NoopRunTimelineMetricsManager = exports.StandardRunTimelineMetricsManager = void 0;
const getEnv_js_1 = require("../utils/getEnv.js");
class StandardRunTimelineMetricsManager {
    _metrics = [];
    registerMetric(metric) {
        this._metrics.push(metric);
    }
    getMetrics() {
        return this._metrics;
    }
    registerMetricsFromExecution(metrics, isWarmStartOverride) {
        this.#seedMetricsFromEnvironment(isWarmStartOverride);
        if (metrics) {
            metrics.forEach((metric) => {
                this.registerMetric({
                    name: `trigger.dev/${metric.name}`,
                    event: metric.event,
                    timestamp: metric.timestamp,
                    attributes: {
                        duration: metric.duration,
                    },
                });
            });
        }
    }
    reset() {
        this._metrics = [];
    }
    // TODO: handle this when processKeepAlive is enabled
    #seedMetricsFromEnvironment(isWarmStartOverride) {
        const forkStartTime = (0, getEnv_js_1.getEnvVar)("TRIGGER_PROCESS_FORK_START_TIME");
        const warmStart = (0, getEnv_js_1.getEnvVar)("TRIGGER_WARM_START");
        const isWarmStart = typeof isWarmStartOverride === "boolean" ? isWarmStartOverride : warmStart === "true";
        if (typeof forkStartTime === "string" && !isWarmStart) {
            const forkStartTimeMs = parseInt(forkStartTime, 10);
            this.registerMetric({
                name: "trigger.dev/start",
                event: "fork",
                attributes: {
                    duration: Date.now() - forkStartTimeMs,
                },
                timestamp: forkStartTimeMs,
            });
        }
    }
}
exports.StandardRunTimelineMetricsManager = StandardRunTimelineMetricsManager;
class NoopRunTimelineMetricsManager {
    registerMetric(metric) {
        // Do nothing
    }
    getMetrics() {
        return [];
    }
}
exports.NoopRunTimelineMetricsManager = NoopRunTimelineMetricsManager;
//# sourceMappingURL=runTimelineMetricsManager.js.map