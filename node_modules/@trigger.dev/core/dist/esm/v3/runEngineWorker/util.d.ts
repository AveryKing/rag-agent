/** Will ignore headers with falsey values */
export declare function createHeaders(headersInit: Record<string, string | undefined | null>): {
    [k: string]: string;
};
