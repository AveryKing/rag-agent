import { createIdempotencyKey, resetIdempotencyKey, type IdempotencyKey } from "@trigger.dev/core/v3";
export declare const idempotencyKeys: {
    create: typeof createIdempotencyKey;
    reset: typeof resetIdempotencyKey;
};
export type { IdempotencyKey };
