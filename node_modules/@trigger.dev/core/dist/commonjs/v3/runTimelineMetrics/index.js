"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunTimelineMetricsAPI = void 0;
const globals_js_1 = require("../utils/globals.js");
const runTimelineMetricsManager_js_1 = require("./runTimelineMetricsManager.js");
const flattenAttributes_js_1 = require("../utils/flattenAttributes.js");
const semanticInternalAttributes_js_1 = require("../semanticInternalAttributes.js");
const API_NAME = "run-timeline-metrics";
const NOOP_MANAGER = new runTimelineMetricsManager_js_1.NoopRunTimelineMetricsManager();
class RunTimelineMetricsAPI {
    static _instance;
    constructor() { }
    static getInstance() {
        if (!this._instance) {
            this._instance = new RunTimelineMetricsAPI();
        }
        return this._instance;
    }
    registerMetric(metric) {
        this.#getManager().registerMetric(metric);
    }
    getMetrics() {
        return this.#getManager().getMetrics();
    }
    /**
     * Measures the execution time of an async function and registers it as a metric
     * @param metricName The name of the metric
     * @param eventName The event name
     * @param attributesOrCallback Optional attributes or the callback function
     * @param callbackFn The async function to measure (if attributes were provided)
     * @returns The result of the callback function
     */
    async measureMetric(metricName, eventName, attributesOrCallback, callbackFn) {
        // Handle overloaded function signature
        let attributes = {};
        let callback;
        if (typeof attributesOrCallback === "function") {
            callback = attributesOrCallback;
        }
        else {
            attributes = attributesOrCallback || {};
            if (!callbackFn) {
                throw new Error("Callback function is required when attributes are provided");
            }
            callback = callbackFn;
        }
        // Record start time
        const startTime = Date.now();
        try {
            // Execute the callback
            const result = await callback();
            // Calculate duration
            const duration = Date.now() - startTime;
            // Register the metric
            this.registerMetric({
                name: metricName,
                event: eventName,
                attributes: {
                    ...attributes,
                    duration,
                },
                timestamp: startTime,
            });
            return result;
        }
        catch (error) {
            // Register the metric even if there's an error, but mark it as failed
            const duration = Date.now() - startTime;
            this.registerMetric({
                name: metricName,
                event: eventName,
                attributes: {
                    ...attributes,
                    duration,
                    error: error instanceof Error ? error.message : String(error),
                    status: "failed",
                },
                timestamp: startTime,
            });
            // Re-throw the error
            throw error;
        }
    }
    convertMetricsToSpanEvents() {
        const metrics = this.getMetrics();
        const spanEvents = metrics.map((metric) => {
            return {
                name: metric.name,
                startTime: metric.timestamp,
                attributes: {
                    ...metric.attributes,
                    event: metric.event,
                },
            };
        });
        return spanEvents;
    }
    convertMetricsToSpanAttributes() {
        const metrics = this.getMetrics();
        if (metrics.length === 0) {
            return {};
        }
        // Group metrics by name
        const metricsByName = metrics.reduce((acc, metric) => {
            if (!acc[metric.name]) {
                acc[metric.name] = [];
            }
            acc[metric.name].push(metric);
            return acc;
        }, {});
        // Process each metric type
        const reducedMetrics = metrics.reduce((acc, metric) => {
            acc[metric.event] = {
                name: metric.name,
                timestamp: metric.timestamp,
                event: metric.event,
                ...(0, flattenAttributes_js_1.flattenAttributes)(metric.attributes, "attributes"),
            };
            return acc;
        }, {});
        const metricEventRollups = {};
        // Calculate duration for each metric type
        // Calculate duration for each metric type
        for (const [metricName, metricEvents] of Object.entries(metricsByName)) {
            // Skip if there are no events for this metric
            if (metricEvents.length === 0)
                continue;
            // Sort events by timestamp
            const sortedEvents = [...metricEvents].sort((a, b) => a.timestamp - b.timestamp);
            // Get first event timestamp (we know it exists because we checked length above)
            const firstTimestamp = sortedEvents[0].timestamp;
            // Get last event (we know it exists because we checked length above)
            const lastEvent = sortedEvents[sortedEvents.length - 1];
            // Calculate total duration: from first event to (last event + its duration)
            // Use optional chaining and nullish coalescing for safety
            const lastEventDuration = lastEvent.attributes?.duration ?? 0;
            const lastEventEndTime = lastEvent.timestamp + lastEventDuration;
            // Store the total duration for this metric type
            const duration = lastEventEndTime - firstTimestamp;
            const timestamp = firstTimestamp;
            metricEventRollups[metricName] = {
                name: metricName,
                duration,
                timestamp,
            };
        }
        return {
            ...(0, flattenAttributes_js_1.flattenAttributes)(reducedMetrics, semanticInternalAttributes_js_1.SemanticInternalAttributes.METRIC_EVENTS),
            ...(0, flattenAttributes_js_1.flattenAttributes)(metricEventRollups, semanticInternalAttributes_js_1.SemanticInternalAttributes.METRIC_EVENTS),
        };
    }
    setGlobalManager(manager) {
        return (0, globals_js_1.registerGlobal)(API_NAME, manager);
    }
    #getManager() {
        return (0, globals_js_1.getGlobal)(API_NAME) ?? NOOP_MANAGER;
    }
}
exports.RunTimelineMetricsAPI = RunTimelineMetricsAPI;
//# sourceMappingURL=index.js.map