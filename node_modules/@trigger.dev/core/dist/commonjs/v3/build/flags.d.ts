/**
 * Deduplicates command line flags by keeping only the last occurrence of each flag.
 * Preserves the order of the last occurrence of each flag.
 *
 * @param flags - A space-separated string of command line flags
 * @returns A space-separated string of deduplicated flags
 *
 * @example
 * // Single flags are preserved
 * dedupFlags("--quiet --verbose") // returns "--quiet --verbose"
 *
 * @example
 * // For duplicate flags, the last value wins and maintains its position
 * dedupFlags("--debug=false --log=info --debug=true") // returns "--log=info --debug=true"
 *
 * @example
 * // Mixing flags with and without values
 * dedupFlags("-v --log=debug -v") // returns "--log=debug -v"
 */
export declare function dedupFlags(flags: string): string;
