"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.list = list;
exports.retrieve = retrieve;
exports.pause = pause;
exports.overrideConcurrencyLimit = overrideConcurrencyLimit;
exports.resetConcurrencyLimit = resetConcurrencyLimit;
exports.resume = resume;
const v3_1 = require("@trigger.dev/core/v3");
const tracer_js_1 = require("./tracer.js");
/**
 * Lists queues
 * @param options - The list options
 * @param options.page - The page number
 * @param options.perPage - The number of queues per page
 * @returns The list of queues
 */
function list(options, requestOptions) {
    const apiClient = v3_1.apiClientManager.clientOrThrow();
    const $requestOptions = (0, v3_1.mergeRequestOptions)({
        tracer: tracer_js_1.tracer,
        name: "queues.list()",
        icon: "queue",
    }, requestOptions);
    return apiClient.listQueues(options, $requestOptions);
}
/**
 * When retrieving a queue you can either use the queue id,
 * or the type and name.
 *
 * @example
 *
 * ```ts
 * // Use a queue id (they start with queue_
 * const q1 = await queues.retrieve("queue_12345");
 *
 * // Or use the type and name
 * // The default queue for your "my-task-id"
 * const q2 = await queues.retrieve({ type: "task", name: "my-task-id"});
 *
 * // The custom queue you defined in your code
 * const q3 = await queues.retrieve({ type: "custom", name: "my-custom-queue" });
 * ```
 * @param queue - The ID of the queue to retrieve, or the type and name
 * @returns The retrieved queue
 */
function retrieve(queue, requestOptions) {
    const apiClient = v3_1.apiClientManager.clientOrThrow();
    const $requestOptions = (0, v3_1.mergeRequestOptions)({
        tracer: tracer_js_1.tracer,
        name: "queues.retrieve()",
        icon: "queue",
        attributes: {
            ...(0, v3_1.flattenAttributes)({ queue }),
            ...(0, v3_1.accessoryAttributes)({
                items: [
                    {
                        text: typeof queue === "string" ? queue : queue.name,
                        variant: "normal",
                    },
                ],
                style: "codepath",
            }),
        },
    }, requestOptions);
    return apiClient.retrieveQueue(queue, $requestOptions);
}
/**
 * Pauses a queue, preventing any new runs from being started.
 * Runs that are currently running will continue to completion.
 *
 * @example
 * ```ts
 * // Pause using a queue id
 * await queues.pause("queue_12345");
 *
 * // Or pause using type and name
 * await queues.pause({ type: "task", name: "my-task-id"});
 * ```
 * @param queue - The ID of the queue to pause, or the type and name
 * @returns The updated queue state
 */
function pause(queue, requestOptions) {
    const apiClient = v3_1.apiClientManager.clientOrThrow();
    const $requestOptions = (0, v3_1.mergeRequestOptions)({
        tracer: tracer_js_1.tracer,
        name: "queues.pause()",
        icon: "queue",
        attributes: {
            ...(0, v3_1.flattenAttributes)({ queue }),
            ...(0, v3_1.accessoryAttributes)({
                items: [
                    {
                        text: typeof queue === "string" ? queue : queue.name,
                        variant: "normal",
                    },
                ],
                style: "codepath",
            }),
        },
    }, requestOptions);
    return apiClient.pauseQueue(queue, "pause", $requestOptions);
}
/**
 * Overrides the concurrency limit of a queue.
 *
 * @param queue - The ID of the queue to override the concurrency limit, or the type and name
 * @param concurrencyLimit - The concurrency limit to override
 * @returns The updated queue state
 */
function overrideConcurrencyLimit(queue, concurrencyLimit, requestOptions) {
    const apiClient = v3_1.apiClientManager.clientOrThrow();
    const $requestOptions = (0, v3_1.mergeRequestOptions)({
        tracer: tracer_js_1.tracer,
        name: "queues.overrideConcurrencyLimit()",
        icon: "queue",
        attributes: {
            ...(0, v3_1.flattenAttributes)({ queue }),
            ...(0, v3_1.accessoryAttributes)({
                items: [
                    {
                        text: typeof queue === "string" ? queue : queue.name,
                        variant: "normal",
                    },
                ],
                style: "codepath",
            }),
        },
    }, requestOptions);
    return apiClient.overrideQueueConcurrencyLimit(queue, concurrencyLimit, $requestOptions);
}
/**
 * Resets the concurrency limit of a queue to the base value.
 *
 * @param queue - The ID of the queue to reset the concurrency limit, or the type and name
 * @returns The updated queue state
 */
function resetConcurrencyLimit(queue, requestOptions) {
    const apiClient = v3_1.apiClientManager.clientOrThrow();
    const $requestOptions = (0, v3_1.mergeRequestOptions)({
        tracer: tracer_js_1.tracer,
        name: "queues.resetConcurrencyLimit()",
        icon: "queue",
        attributes: {
            ...(0, v3_1.flattenAttributes)({ queue }),
            ...(0, v3_1.accessoryAttributes)({
                items: [
                    {
                        text: typeof queue === "string" ? queue : queue.name,
                        variant: "normal",
                    },
                ],
                style: "codepath",
            }),
        },
    }, requestOptions);
    return apiClient.resetQueueConcurrencyLimit(queue, $requestOptions);
}
/**
 * Resumes a paused queue, allowing new runs to be started.
 *
 * @example
 * ```ts
 * // Resume using a queue id
 * await queues.resume("queue_12345");
 *
 * // Or resume using type and name
 * await queues.resume({ type: "task", name: "my-task-id"});
 * ```
 * @param queue - The ID of the queue to resume, or the type and name
 * @returns The updated queue state
 */
function resume(queue, requestOptions) {
    const apiClient = v3_1.apiClientManager.clientOrThrow();
    const $requestOptions = (0, v3_1.mergeRequestOptions)({
        tracer: tracer_js_1.tracer,
        name: "queues.resume()",
        icon: "queue",
        attributes: {
            ...(0, v3_1.flattenAttributes)({ queue }),
            ...(0, v3_1.accessoryAttributes)({
                items: [
                    {
                        text: typeof queue === "string" ? queue : queue.name,
                        variant: "normal",
                    },
                ],
                style: "codepath",
            }),
        },
    }, requestOptions);
    return apiClient.pauseQueue(queue, "resume", $requestOptions);
}
//# sourceMappingURL=queues.js.map