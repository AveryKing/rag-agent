/**
 * Attempts to parse a string into a valid Date.
 *
 * Supported formats:
 * - ISO and RFC date strings (e.g. "2025-08-18", "2025-08-18T12:34:56Z")
 * - Natural language dates supported by JS Date (e.g. "August 18, 2025")
 * - Epoch seconds (10-digit numeric string, e.g. "1629302400")
 * - Epoch milliseconds (13-digit numeric string, e.g. "1629302400000")
 *
 * @param input The string to parse.
 * @returns A valid Date object, or undefined if parsing fails.
 */
export declare function parseDate(input: string): Date | undefined;
