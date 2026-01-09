export function createAsyncIterableStream(source, transformer) {
    const transformedStream = source.pipeThrough(new TransformStream(transformer));
    transformedStream[Symbol.asyncIterator] = () => {
        const reader = transformedStream.getReader();
        return {
            async next() {
                const { done, value } = await reader.read();
                return done ? { done: true, value: undefined } : { done: false, value };
            },
        };
    };
    return transformedStream;
}
export function createAsyncIterableReadable(source, transformer, signal) {
    return new ReadableStream({
        async start(controller) {
            const transformedStream = source.pipeThrough(new TransformStream(transformer));
            const reader = transformedStream.getReader();
            signal.addEventListener("abort", () => {
                queueMicrotask(() => {
                    reader.cancel();
                    controller.close();
                });
            });
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    controller.close();
                    break;
                }
                controller.enqueue(value);
            }
        },
    });
}
export function createAsyncIterableStreamFromAsyncIterable(asyncIterable, transformer, signal) {
    const stream = new ReadableStream({
        async start(controller) {
            try {
                if (signal) {
                    signal.addEventListener("abort", () => {
                        controller.close();
                    });
                }
                const iterator = asyncIterable[Symbol.asyncIterator]();
                while (true) {
                    if (signal?.aborted) {
                        break;
                    }
                    const { done, value } = await iterator.next();
                    if (done) {
                        controller.close();
                        break;
                    }
                    controller.enqueue(value);
                }
            }
            catch (error) {
                controller.error(error);
            }
        },
        cancel() {
            // If the stream is a tinyexec process with a kill method, kill it
            if ("kill" in asyncIterable) {
                asyncIterable.kill();
            }
        },
    });
    const transformedStream = stream.pipeThrough(new TransformStream(transformer));
    return transformedStream;
}
export function createAsyncIterableStreamFromAsyncGenerator(asyncGenerator, transformer, signal) {
    return createAsyncIterableStreamFromAsyncIterable(asyncGenerator, transformer, signal);
}
export function ensureAsyncIterable(input) {
    // If it's already an AsyncIterable, return it as-is
    if (Symbol.asyncIterator in input) {
        return input;
    }
    // Convert ReadableStream to AsyncIterable
    const readableStream = input;
    return {
        async *[Symbol.asyncIterator]() {
            const reader = readableStream.getReader();
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    if (value !== undefined) {
                        yield value;
                    }
                }
            }
            finally {
                reader.releaseLock();
            }
        },
    };
}
export function ensureReadableStream(input) {
    if ("getReader" in input) {
        return input;
    }
    return new ReadableStream({
        async start(controller) {
            const iterator = input[Symbol.asyncIterator]();
            while (true) {
                const { done, value } = await iterator.next();
                if (done) {
                    break;
                }
                controller.enqueue(value);
            }
            controller.close();
        },
    });
}
//# sourceMappingURL=asyncIterableStream.js.map