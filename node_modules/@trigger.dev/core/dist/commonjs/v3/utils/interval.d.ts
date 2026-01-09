type IntervalServiceOptions = {
    onInterval: () => Promise<void>;
    onError?: (error: unknown) => Promise<void>;
    intervalMs?: number;
    leadingEdge?: boolean;
};
export declare class IntervalService {
    #private;
    private _onInterval;
    private _onError?;
    private _intervalMs;
    private _nextInterval;
    private _leadingEdge;
    private _isEnabled;
    private _isExecuting;
    constructor(opts: IntervalServiceOptions);
    start(): void;
    stop(): {
        isExecuting: boolean;
    };
    resetCurrentInterval(): void;
    updateInterval(intervalMs: number): void;
    get intervalMs(): number;
}
export {};
