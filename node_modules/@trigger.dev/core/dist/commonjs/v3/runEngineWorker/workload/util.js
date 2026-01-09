"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getDefaultWorkloadHeaders = getDefaultWorkloadHeaders;
const consts_js_1 = require("../consts.js");
const util_js_1 = require("../util.js");
function getDefaultWorkloadHeaders(options) {
    return (0, util_js_1.createHeaders)({
        [consts_js_1.WORKLOAD_HEADERS.DEPLOYMENT_ID]: options.deploymentId,
        [consts_js_1.WORKLOAD_HEADERS.RUNNER_ID]: options.runnerId,
        [consts_js_1.WORKLOAD_HEADERS.DEPLOYMENT_VERSION]: options.deploymentVersion,
        [consts_js_1.WORKLOAD_HEADERS.PROJECT_REF]: options.projectRef,
    });
}
//# sourceMappingURL=util.js.map