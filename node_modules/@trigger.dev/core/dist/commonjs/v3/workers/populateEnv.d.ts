/**
 * Options for environment variable population
 */
interface PopulateEnvOptions {
    /**
     * Whether to override existing environment variables
     * @default false
     */
    override?: boolean;
    /**
     * Whether to enable debug logging
     * @default false
     */
    debug?: boolean;
    /**
     * The previous environment variables
     * @default undefined
     */
    previousEnv?: Record<string, string>;
}
/**
 * Populates process.env with values from the provided object
 *
 * @param envObject - Object containing environment variables to set
 * @param options - Optional configuration
 */
export declare function populateEnv(envObject: Record<string, string>, options?: PopulateEnvOptions): void;
export {};
