import { type PlacementTag } from "../schemas/index.js";
export interface PlacementConfig {
    enabled: boolean;
    prefix: string;
}
export declare class PlacementTagProcessor {
    private readonly config;
    private readonly logger;
    constructor(config: PlacementConfig);
    /**
     * Converts placement tags to Kubernetes nodeSelector labels
     */
    convertToNodeSelector(placementTags?: PlacementTag[], existingNodeSelector?: Record<string, string>): Record<string, string>;
    private printTagWarnings;
}
/**
 * Helper function to create a placement tag. In the future this will be able to support multiple values and operators.
 * For now it's just a single value.
 */
export declare function placementTag(key: string, value: string): PlacementTag;
