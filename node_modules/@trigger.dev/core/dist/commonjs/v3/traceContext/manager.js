"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StandardTraceContextManager = void 0;
const api_1 = require("@opentelemetry/api");
const core_1 = require("@opentelemetry/core");
class StandardTraceContextManager {
    traceContext = {};
    getTraceContext() {
        return this.traceContext;
    }
    reset() {
        this.traceContext = {};
    }
    getExternalTraceContext() {
        return extractExternalTraceContext(this.traceContext?.external);
    }
    extractContext() {
        return api_1.propagation.extract(api_1.context.active(), this.traceContext ?? {});
    }
    withExternalTrace(fn) {
        const externalTraceContext = this.getExternalTraceContext();
        if (!externalTraceContext) {
            return fn();
        }
        // Get the current active span context to extract the span ID
        const currentSpanContext = api_1.trace.getActiveSpan()?.spanContext();
        if (!currentSpanContext) {
            throw new Error("No active span found. withExternalSpan must be called within an active span context.");
        }
        const spanContext = {
            traceId: externalTraceContext.traceId,
            spanId: currentSpanContext.spanId,
            traceFlags: externalTraceContext.traceFlags,
            isRemote: true,
        };
        const contextWithSpan = api_1.trace.setSpanContext(api_1.context.active(), spanContext);
        return api_1.context.with(contextWithSpan, fn);
    }
}
exports.StandardTraceContextManager = StandardTraceContextManager;
function extractExternalTraceContext(traceContext) {
    if (typeof traceContext !== "object" || traceContext === null) {
        return undefined;
    }
    const tracestate = "tracestate" in traceContext && typeof traceContext.tracestate === "string"
        ? traceContext.tracestate
        : undefined;
    if ("traceparent" in traceContext && typeof traceContext.traceparent === "string") {
        const externalSpanContext = (0, core_1.parseTraceParent)(traceContext.traceparent);
        if (!externalSpanContext) {
            return undefined;
        }
        return {
            traceId: externalSpanContext.traceId,
            spanId: externalSpanContext.spanId,
            traceFlags: externalSpanContext.traceFlags,
            tracestate: tracestate,
        };
    }
    return undefined;
}
//# sourceMappingURL=manager.js.map