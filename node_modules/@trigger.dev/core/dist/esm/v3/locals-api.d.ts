import { LocalsAPI } from "./locals/index.js";
import type { LocalsKey } from "./locals/types.js";
/** Entrypoint for runtime API */
export declare const localsAPI: LocalsAPI;
export declare const locals: {
    create<T>(id: string): LocalsKey<T>;
    get<T>(key: LocalsKey<T>): T | undefined;
    getOrThrow<T>(key: LocalsKey<T>): T;
    set<T>(key: LocalsKey<T>, value: T): T;
};
export type Locals = typeof locals;
export type { LocalsKey };
