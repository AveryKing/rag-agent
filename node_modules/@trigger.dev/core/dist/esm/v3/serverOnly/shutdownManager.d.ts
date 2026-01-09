type ShutdownHandler = NodeJS.SignalsListener;
type ShutdownSignal = Extract<NodeJS.Signals, "SIGTERM" | "SIGINT">;
export declare class ShutdownManager {
    private disableForTesting;
    private isShuttingDown;
    private signalNumbers;
    private logger;
    private handlers;
    constructor(disableForTesting?: boolean);
    register(name: string, handler: ShutdownHandler, signals?: ShutdownSignal[]): void;
    unregister(name: string): void;
    shutdown(signal: ShutdownSignal): Promise<void>;
    private isEnabled;
    _getHandlersForTesting(): ReadonlyMap<string, {
        handler: ShutdownHandler;
        signals: ShutdownSignal[];
    }>;
}
export declare const shutdownManager: ShutdownManager;
export {};
