import type * as logsAPI from "@opentelemetry/api-logs";
export declare class ConsoleInterceptor {
    #private;
    private readonly logger;
    private readonly sendToStdIO;
    private readonly interceptingDisabled;
    private readonly maxAttributeCount?;
    constructor(logger: logsAPI.Logger, sendToStdIO: boolean, interceptingDisabled: boolean, maxAttributeCount?: number | undefined);
    intercept<T>(console: Console, callback: () => Promise<T>): Promise<T>;
    debug(...args: unknown[]): void;
    log(...args: unknown[]): void;
    info(...args: unknown[]): void;
    warn(...args: unknown[]): void;
    error(...args: unknown[]): void;
}
