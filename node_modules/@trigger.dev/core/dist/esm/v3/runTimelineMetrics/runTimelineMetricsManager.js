import { getEnvVar } from "../utils/getEnv.js";
export class StandardRunTimelineMetricsManager {
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
        const forkStartTime = getEnvVar("TRIGGER_PROCESS_FORK_START_TIME");
        const warmStart = getEnvVar("TRIGGER_WARM_START");
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
export class NoopRunTimelineMetricsManager {
    registerMetric(metric) {
        // Do nothing
    }
    getMetrics() {
        return [];
    }
}
//# sourceMappingURL=runTimelineMetricsManager.js.map