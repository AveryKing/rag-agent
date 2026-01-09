import { type Span } from "@opentelemetry/api";
export declare function recordSpanException(span: Span, error: unknown): void;
export declare function carrierFromContext(): Record<string, string>;
