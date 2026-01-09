const MINIMUM_MAX_DURATION = 5;
const MAXIMUM_MAX_DURATION = 2_147_483_647; // largest 32-bit signed integer
export function clampMaxDuration(maxDuration) {
    return Math.min(Math.max(maxDuration, MINIMUM_MAX_DURATION), MAXIMUM_MAX_DURATION);
}
export function getMaxDuration(maxDuration, defaultMaxDuration) {
    if (!maxDuration) {
        return defaultMaxDuration ?? undefined;
    }
    // Setting the maxDuration to MAXIMUM_MAX_DURATION means we don't want to use the default maxDuration
    if (maxDuration === MAXIMUM_MAX_DURATION) {
        return;
    }
    return maxDuration;
}
//# sourceMappingURL=maxDuration.js.map