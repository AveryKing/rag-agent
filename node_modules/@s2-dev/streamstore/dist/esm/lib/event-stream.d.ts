export type SseMessage<T> = {
    data?: T | undefined;
    event?: string | undefined;
    id?: string | undefined;
    retry?: number | undefined;
};
type ParseResult<T> = {
    done: true;
    value?: undefined;
    batch?: undefined;
} | {
    done: false;
    batch?: false;
    value?: T;
} | {
    done: false;
    batch: true;
    value: T[];
};
export declare class EventStream<T> extends ReadableStream<T> implements AsyncDisposable {
    constructor(responseBody: ReadableStream<Uint8Array>, parse: (x: SseMessage<string>) => ParseResult<T>);
    [Symbol.asyncDispose](): Promise<void>;
    [Symbol.asyncIterator](): AsyncIterableIterator<T>;
}
export {};
//# sourceMappingURL=event-stream.d.ts.map