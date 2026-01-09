import { WORKLOAD_HEADERS } from "../consts.js";
import { createHeaders } from "../util.js";
export function getDefaultWorkloadHeaders(options) {
    return createHeaders({
        [WORKLOAD_HEADERS.DEPLOYMENT_ID]: options.deploymentId,
        [WORKLOAD_HEADERS.RUNNER_ID]: options.runnerId,
        [WORKLOAD_HEADERS.DEPLOYMENT_VERSION]: options.deploymentVersion,
        [WORKLOAD_HEADERS.PROJECT_REF]: options.projectRef,
    });
}
//# sourceMappingURL=util.js.map