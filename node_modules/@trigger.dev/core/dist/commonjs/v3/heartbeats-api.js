"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.heartbeats = void 0;
// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
const api_js_1 = require("./heartbeats/api.js");
/** Entrypoint for heartbeats API */
exports.heartbeats = api_js_1.HeartbeatsAPI.getInstance();
//# sourceMappingURL=heartbeats-api.js.map