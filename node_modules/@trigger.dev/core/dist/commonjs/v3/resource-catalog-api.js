"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.resourceCatalog = void 0;
// Split module-level variable definition into separate files to allow
// tree-shaking on each api instance.
const index_js_1 = require("./resource-catalog/index.js");
/** Entrypoint for runtime API */
exports.resourceCatalog = index_js_1.ResourceCatalogAPI.getInstance();
//# sourceMappingURL=resource-catalog-api.js.map