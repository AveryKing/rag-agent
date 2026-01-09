"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.lifecycleHooks = void 0;
// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
const index_js_1 = require("./lifecycleHooks/index.js");
/** Entrypoint for runtime API */
exports.lifecycleHooks = index_js_1.LifecycleHooksAPI.getInstance();
//# sourceMappingURL=lifecycle-hooks-api.js.map