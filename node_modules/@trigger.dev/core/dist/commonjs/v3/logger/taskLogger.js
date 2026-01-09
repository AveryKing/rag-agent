"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NoopTaskLogger = exports.OtelTaskLogger = exports.logLevels = void 0;
const api_logs_1 = require("@opentelemetry/api-logs");
const icons_js_1 = require("../icons.js");
const semanticInternalAttributes_js_1 = require("../semanticInternalAttributes.js");
const flattenAttributes_js_1 = require("../utils/flattenAttributes.js");
const clock_api_js_1 = require("../clock-api.js");
exports.logLevels = ["none", "error", "warn", "info", "debug"];
class OtelTaskLogger {
    _config;
    _level;
    constructor(_config) {
        this._config = _config;
        this._level = exports.logLevels.indexOf(_config.level);
    }
    debug(message, properties) {
        if (this._level < 4)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "debug", api_logs_1.SeverityNumber.DEBUG, properties);
    }
    log(message, properties) {
        if (this._level < 3)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "log", api_logs_1.SeverityNumber.INFO, properties);
    }
    info(message, properties) {
        if (this._level < 3)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "info", api_logs_1.SeverityNumber.INFO, properties);
    }
    warn(message, properties) {
        if (this._level < 2)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "warn", api_logs_1.SeverityNumber.WARN, properties);
    }
    error(message, properties) {
        if (this._level < 1)
            return; // ["none", "error", "warn", "info", "debug"];
        this.#emitLog(message, this.#getTimestampInHrTime(), "error", api_logs_1.SeverityNumber.ERROR, properties);
    }
    #emitLog(message, timestamp, severityText, severityNumber, properties) {
        let attributes = {};
        if (properties) {
            // Use flattenAttributes directly - it now handles all non-JSON friendly values efficiently
            attributes = (0, flattenAttributes_js_1.flattenAttributes)(properties, undefined, this._config.maxAttributeCount);
        }
        const icon = (0, icons_js_1.iconStringForSeverity)(severityNumber);
        if (icon !== undefined) {
            attributes[semanticInternalAttributes_js_1.SemanticInternalAttributes.STYLE_ICON] = icon;
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
                [semanticInternalAttributes_js_1.SemanticInternalAttributes.STYLE_ICON]: options?.icon ?? "trace",
            },
        };
        return this._config.tracer.startActiveSpan(name, fn, spanOptions);
    }
    startSpan(name, options) {
        const spanOptions = {
            ...options,
            attributes: {
                ...options?.attributes,
                ...(options?.icon ? { [semanticInternalAttributes_js_1.SemanticInternalAttributes.STYLE_ICON]: options.icon } : {}),
            },
        };
        return this._config.tracer.startSpan(name, spanOptions);
    }
    #getTimestampInHrTime() {
        return clock_api_js_1.clock.preciseNow();
    }
}
exports.OtelTaskLogger = OtelTaskLogger;
class NoopTaskLogger {
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
exports.NoopTaskLogger = NoopTaskLogger;
//# sourceMappingURL=taskLogger.js.map