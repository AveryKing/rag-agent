import { getEnvVar } from "./utils/getEnv.js";
function getOtelEnvVarLimit(key, defaultValue) {
    const value = getEnvVar(key);
    if (!value) {
        return defaultValue;
    }
    return parseInt(value, 10);
}
export const OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", 1024);
export const OTEL_LOG_ATTRIBUTE_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_LOG_ATTRIBUTE_COUNT_LIMIT", 1024);
export const OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", 131072);
export const OTEL_LOG_ATTRIBUTE_VALUE_LENGTH_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_LOG_ATTRIBUTE_VALUE_LENGTH_LIMIT", 131072);
export const OTEL_SPAN_EVENT_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_SPAN_EVENT_COUNT_LIMIT", 10);
export const OTEL_LINK_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_LINK_COUNT_LIMIT", 2);
export const OTEL_ATTRIBUTE_PER_LINK_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_ATTRIBUTE_PER_LINK_COUNT_LIMIT", 10);
export const OTEL_ATTRIBUTE_PER_EVENT_COUNT_LIMIT = getOtelEnvVarLimit("TRIGGER_OTEL_ATTRIBUTE_PER_EVENT_COUNT_LIMIT", 10);
export const OFFLOAD_IO_PACKET_LENGTH_LIMIT = 128 * 1024;
export function imposeAttributeLimits(attributes) {
    const newAttributes = {};
    for (const [key, value] of Object.entries(attributes)) {
        if (calculateAttributeValueLength(value) > OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT) {
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