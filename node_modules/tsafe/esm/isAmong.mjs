/** https://docs.tsafe.dev/isamong */
export function isAmong(names, value) {
    for (const name of names) {
        if (name === value) {
            return true;
        }
    }
    return false;
}
//# sourceMappingURL=isAmong.mjs.map