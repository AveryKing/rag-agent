export interface TimeoutManager {
    abortAfterTimeout: (timeoutInSeconds?: number) => AbortController;
    signal?: AbortSignal;
    reset: () => void;
    registerListener?: (listener: (timeoutInSeconds: number, elapsedTimeInSeconds: number) => void | Promise<void>) => void;
}
export declare class TaskRunExceededMaxDuration extends Error {
    readonly timeoutInSeconds: number;
    readonly usageInSeconds: number;
    constructor(timeoutInSeconds: number, usageInSeconds: number);
}
