"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.preventMultipleWaits = preventMultipleWaits;
const errors_js_1 = require("../errors.js");
const common_js_1 = require("../schemas/common.js");
const concurrentWaitErrorMessage = "Parallel waits are not supported, e.g. using Promise.all() around our wait functions.";
function preventMultipleWaits() {
    let isExecutingWait = false;
    return async (cb) => {
        if (isExecutingWait) {
            console.error(concurrentWaitErrorMessage);
            throw new errors_js_1.InternalError({
                code: common_js_1.TaskRunErrorCodes.TASK_DID_CONCURRENT_WAIT,
                message: concurrentWaitErrorMessage,
                skipRetrying: true,
                showStackTrace: false,
            });
        }
        isExecutingWait = true;
        try {
            //delay calling the callback by one tick
            //(to ensure the first wait doesn't checkpoint before the second is called)
            await Promise.resolve();
            return await cb();
        }
        finally {
            isExecutingWait = false;
        }
    };
}
//# sourceMappingURL=preventMultipleWaits.js.map