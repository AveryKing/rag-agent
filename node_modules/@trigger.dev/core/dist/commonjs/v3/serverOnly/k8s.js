"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isKubernetesEnvironment = isKubernetesEnvironment;
const std_env_1 = require("std-env");
function isKubernetesEnvironment(override) {
    if (override !== undefined) {
        return override;
    }
    // Then check for common Kubernetes environment variables
    const k8sIndicators = [
        std_env_1.env.KUBERNETES_PORT,
        std_env_1.env.KUBERNETES_SERVICE_HOST,
        std_env_1.env.KUBERNETES_SERVICE_PORT,
    ];
    console.debug("k8sIndicators", { k8sIndicators });
    return k8sIndicators.some(Boolean);
}
//# sourceMappingURL=k8s.js.map