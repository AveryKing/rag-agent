"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.traceContext = void 0;
// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
const api_js_1 = require("./traceContext/api.js");
/** Entrypoint for trace context API */
exports.traceContext = api_js_1.TraceContextAPI.getInstance();
//# sourceMappingURL=trace-context-api.js.map