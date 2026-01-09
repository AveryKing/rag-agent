import { SimpleStructuredLogger } from "../utils/structuredLogger.js";
export class PlacementTagProcessor {
    config;
    logger = new SimpleStructuredLogger("placement-tag-processor");
    constructor(config) {
        this.config = config;
    }
    /**
     * Converts placement tags to Kubernetes nodeSelector labels
     */
    convertToNodeSelector(placementTags, existingNodeSelector) {
        if (!this.config.enabled || !placementTags || placementTags.length === 0) {
            return existingNodeSelector ?? {};
        }
        const nodeSelector = { ...existingNodeSelector };
        // Convert placement tags to nodeSelector labels
        for (const tag of placementTags) {
            const labelKey = `${this.config.prefix}/${tag.key}`;
            // Print warnings (if any)
            this.printTagWarnings(tag);
            // For now we only support single values via nodeSelector
            nodeSelector[labelKey] = tag.values?.[0] ?? "";
        }
        return nodeSelector;
    }
    printTagWarnings(tag) {
        if (!tag.values || tag.values.length === 0) {
            // No values provided
            this.logger.warn("Placement tag has no values, using empty string", tag);
        }
        else if (tag.values.length > 1) {
            // Multiple values provided
            this.logger.warn("Placement tag has multiple values, only using first one", tag);
        }
    }
}
/**
 * Helper function to create a placement tag. In the future this will be able to support multiple values and operators.
 * For now it's just a single value.
 */
export function placementTag(key, value) {
    return { key, values: [value] };
}
//# sourceMappingURL=placementTags.js.map