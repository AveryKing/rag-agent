export type AsyncIterableStream<T> = AsyncIterable<T> & ReadableStream<T>;
export declare function createAsyncIterableStream<S, T>(source: ReadableStream<S>, transformer: Transformer<S, T>): AsyncIterableStream<T>;
export declare function createAsyncIterableReadable<S, T>(source: ReadableStream<S>, transformer: Transformer<S, T>, signal: AbortSignal): AsyncIterableStream<T>;
export declare function createAsyncIterableStreamFromAsyncIterable<T>(asyncIterable: AsyncIterable<T>, transformer?: Transformer<T, T>, signal?: AbortSignal): AsyncIterableStream<T>;
export declare function createAsyncIterableStreamFromAsyncGenerator<T>(asyncGenerator: AsyncGenerator<T, void, unknown>, transformer: Transformer<T, T>, signal?: AbortSignal): AsyncIterableStream<T>;
export declare function ensureAsyncIterable<T>(input: AsyncIterable<T> | ReadableStream<T>): AsyncIterable<T>;
export declare function ensureReadableStream<T>(input: AsyncIterable<T> | ReadableStream<T>): ReadableStream<T>;
