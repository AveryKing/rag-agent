"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getEnvVar = getEnvVar;
exports.getNumberEnvVar = getNumberEnvVar;
const std_env_1 = require("std-env");
/**
 * Get an environment variable with optional default value. Runtime agnostic.
 *
 * @param name The name of the environment variable.
 * @param defaultValue The default value to return if the environment variable is not set.
 * @returns The value of the environment variable, or the default value if the environment variable is not set.
 *
 */
function getEnvVar(name, defaultValue) {
    return std_env_1.env[name] ?? defaultValue;
}
function getNumberEnvVar(name, defaultValue) {
    const value = getEnvVar(name);
    if (value === undefined) {
        return defaultValue;
    }
    const parsed = Number(value);
    if (isNaN(parsed)) {
        return defaultValue;
    }
    return parsed;
}
//# sourceMappingURL=getEnv.js.map