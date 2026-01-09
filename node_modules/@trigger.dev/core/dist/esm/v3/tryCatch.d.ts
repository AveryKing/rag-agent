export type Success<T> = [null, T];
export type Failure<E> = [E, null];
export type Result<T, E = Error> = Success<T> | Failure<E>;
export declare function tryCatch<T, E = Error>(promise: Promise<T>): Promise<Result<T, E>>;
