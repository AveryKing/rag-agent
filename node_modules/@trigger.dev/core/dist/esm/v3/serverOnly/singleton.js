export function singleton(name, getValue) {
    const thusly = globalThis;
    thusly.__trigger_singletons ??= {};
    thusly.__trigger_singletons[name] ??= getValue();
    return thusly.__trigger_singletons[name];
}
//# sourceMappingURL=singleton.js.map