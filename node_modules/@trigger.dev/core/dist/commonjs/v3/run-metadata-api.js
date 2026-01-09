"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runMetadata = void 0;
// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
const index_js_1 = require("./runMetadata/index.js");
exports.runMetadata = index_js_1.RunMetadataAPI.getInstance();
__exportStar(require("./runMetadata/types.js"), exports);
__exportStar(require("./runMetadata/operations.js"), exports);
//# sourceMappingURL=run-metadata-api.js.map