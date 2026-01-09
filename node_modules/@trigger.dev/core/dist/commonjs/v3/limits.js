"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OFFLOAD_IO_PACKET_LENGTH_LIMIT = exports.OTEL_ATTRIBUTE_PER_EVENT_COUNT_LIMIT = exports.OTEL_ATTRIBUTE_PER_LINK_COUNT_LIMIT = exports.OTEL_LINK_COUNT_LIMIT = exports.OTEL_SPAN_EVENT_COUNT_LIMIT = exports.OTEL_LOG_ATTRIBUTE_VALUE_LENGTH_LIMIT = exports.OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT = exports.OTEL_LOG_ATTRIBUTE_COUNT_LIMIT = exports.OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT = void 0;
exports.imposeAttributeLimits = imposeAttributeLimits;
const getEnv_js_1 = require("./utils/getEnv.js");
function getOtelEnvVarLimit(key, defaultValue) {
    const value = (0, getEnv_js_1.getEnvVar)(key);
    if (!value) {
        return defaultValue;
    }
    return parseInt(value, 10);
}
exports.OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", 1024);
exports.OTEL_LOG_ATTRIBUTE_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_LOG_ATTRIBUTE_COUNT_LIMIT", 1024);
exports.OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", 131072);
exports.OTEL_LOG_ATTRIBUTE_VALUE_LENGTH_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_LOG_ATTRIBUTE_VALUE_LENGTH_LIMIT", 131072);
exports.OTEL_SPAN_EVENT_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_SPAN_EVENT_COUNT_LIMIT", 10);
exports.OTEL_LINK_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_LINK_COUNT_LIMIT", 2);
exports.OTEL_ATTRIBUTE_PER_LINK_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_ATTRIBUTE_PER_LINK_COUNT_LIMIT", 10);
exports.OTEL_ATTRIBUTE_PER_EVENT_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_ATTRIBUTE_PER_EVENT_COUNT_LIMIT", 10);
exports.OFFLOAD_IO_PACKET_LENGTH_LIMIT = 128 * 1024;
function imposeAttributeLimits(attributes) {
    const newAttributes = {};
    for (const [key, value] of Object.entries(attributes)) {
        if (calculateAttributeValueLength(value) > exports.OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT) {
            continue;
        }
        newAttributes[key] = value;
    }
    return newAttributes;
}
function calculateAttributeValueLength(value) {
    if (value === undefined || value === null) {
        return 0;
    }
    if (typeof value === "string") {
        return value.length;
    }
    if (typeof value === "number") {
        return 8;
    }
    if (typeof value === "boolean") {
        return 4;
    }
    if (Array.isArray(value)) {
        return value.reduce((acc, v) => acc + calculateAttributeValueLength(v), 0);
    }
    return 0;
}
//# sourceMappingURL=limits.js.map