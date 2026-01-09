import { Webhook } from "@trigger.dev/core/v3";
/**
 * The type of error thrown when a webhook fails to parse or verify
 */
export declare class WebhookError extends Error {
    constructor(message: string);
}
/**
 * Options for constructing a webhook event
 */
type ConstructEventOptions = {
    /** Raw payload as string or Buffer */
    payload: string | Buffer;
    /** Signature header as string, Buffer, or string array */
    header: string | Buffer | Array<string>;
};
/**
 * Interface describing the webhook utilities
 */
interface Webhooks {
    /**
     * Constructs and validates a webhook event from an incoming request
     * @param request - Either a Request object or ConstructEventOptions containing the payload and signature
     * @param secret - Secret key used to verify the webhook signature
     * @returns Promise resolving to a validated AlertWebhook object
     * @throws {WebhookError} If validation fails or payload can't be parsed
     *
     * @example
     * // Using with Request object
     * const event = await webhooks.constructEvent(request, "webhook_secret");
     *
     * @example
     * // Using with manual options
     * const event = await webhooks.constructEvent({
     *   payload: rawBody,
     *   header: signatureHeader
     * }, "webhook_secret");
     */
    constructEvent(request: ConstructEventOptions | Request, secret: string): Promise<Webhook>;
    /** Header name used for webhook signatures */
    SIGNATURE_HEADER_NAME: string;
}
/**
 * Webhook utilities for handling incoming webhook requests
 */
export declare const webhooks: Webhooks;
export {};
