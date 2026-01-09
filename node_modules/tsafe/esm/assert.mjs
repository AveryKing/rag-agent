/** https://docs.tsafe.dev/assert#error-thrown */
export class AssertionError extends Error {
    constructor(msg) {
        super(`Wrong assertion encountered` + (!msg ? "" : `: "${msg}"`));
        this.originalMessage = msg;
        Object.setPrototypeOf(this, new.target.prototype);
    }
}
let refOfIs = undefined;
/** https://docs.tsafe.dev/assert */
export function assert(condition, msg) {
    if (arguments.length === 0) {
        condition = true;
    }
    if (refOfIs !== undefined) {
        refOfIs = undefined;
        return;
    }
    if (!condition) {
        const error = new AssertionError(typeof msg === "function" ? msg() : msg);
        if (Error.captureStackTrace) {
            Error.captureStackTrace(error, assert);
        }
        throw error;
    }
}
const errorMessage = "Wrong usage of the `is` function refer to https://docs.tsafe.dev/is";
/** https://docs.tsafe.dev/is */
export function is(value) {
    const ref = {};
    if (refOfIs !== undefined) {
        refOfIs = undefined;
        throw new Error(errorMessage);
    }
    refOfIs = ref;
    Promise.resolve().then(() => {
        if (refOfIs === ref) {
            throw new Error(errorMessage);
        }
    });
    return null;
}
//# sourceMappingURL=assert.mjs.map