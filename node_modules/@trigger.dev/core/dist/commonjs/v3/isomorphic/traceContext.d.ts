export declare function parseTraceparent(traceparent?: string): {
    traceId: string;
    spanId: string;
    traceFlags?: string;
} | undefined;
export declare function serializeTraceparent(traceId: string, spanId: string, traceFlags?: string): string;
