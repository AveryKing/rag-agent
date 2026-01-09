import { SeverityNumber } from "@opentelemetry/api-logs";
import { iconStringForSeverity } from "../icons.js";
import { SemanticInternalAttributes } from "../semanticInternalAttributes.js";
import { flattenAttributes } from "../utils/flattenAttributes.js";
import { clock } from "../clock-api.js";
export const logLevels = ["none", "error", "warn", "info", "debug"];
export class OtelTaskLogger {
    _config;
    _level;
    constructor(_config) {
        this._config = _config;
        this._level = logLevels.indexOf(_config.level);
    }
    debug(message, properties) {
        if (this._level < 4)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "debug", SeverityNumber.DEBUG, properties);
    }
    log(message, properties) {
        if (this._level < 3)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "log", SeverityNumber.INFO, properties);
    }
    info(message, properties) {
        if (this._level < 3)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "info", SeverityNumber.INFO, properties);
    }
    warn(message, properties) {
        if (this._level < 2)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "warn", SeverityNumber.WARN, properties);
    }
    error(message, properties) {
        if (this._level < 1)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "error", SeverityNumber.ERROR, properties);
    }
    #emitLog(message, timestamp, severityText, severityNumber, properties) {
        let attributes = {};
        if (properties) {
            // Use flattenAttributes directly - it now handles all non-JSON friendly values efficiently
            attributes = flattenAttributes(properties, undefined, this._config.maxAttributeCount);
        }
        const icon = iconStringForSeverity(severityNumber);
        if (icon !== undefined) {
            attributes[SemanticInternalAttributes.STYLE_ICON] = icon;
        }
        this._config.logger.emit({
            severityNumber,
            severityText,
            body: message,
            attributes,
            timestamp,
        });
    }
    trace(name, fn, options) {
        const spanOptions = {
            ...options,
            attributes: {
                ...options?.attributes,
                [SemanticInternalAttributes.STYLE_ICON]: options?.icon ?? "trace",
            },
        };
        return this._config.tracer.startActiveSpan(name, fn, spanOptions);
    }
    startSpan(name, options) {
        const spanOptions = {
            ...options,
            attributes: {
                ...options?.attributes,
                ...(options?.icon ? { [SemanticInternalAttributes.STYLE_ICON]: options.icon } : {}),
            },
        };
        return this._config.tracer.startSpan(name, spanOptions);
    }
    #getTimestampInHrTime() {
        return clock.preciseNow();
    }
}
export class NoopTaskLogger {
    debug() { }
    log() { }
    info() { }
    warn() { }
    error() { }
    trace(name, fn) {
        return fn({});
    }
    startSpan() {
        return {};
    }
}
//# sourceMappingURL=taskLogger.js.map