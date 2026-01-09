import { SimpleStructuredLogger } from "../../utils/structuredLogger.js";
/**
 * Processes queue length samples using exponential weighted moving average (EWMA)
 * for smoothing and median filtering for outlier resistance.
 *
 * Collects samples within a batch window, calculates median to filter outliers,
 * then applies EWMA smoothing for stable trend tracking.
 */
export class QueueMetricsProcessor {
    ewmaAlpha;
    batchWindowMs;
    logger = new SimpleStructuredLogger("queue-metrics-processor");
    samples = [];
    smoothedValue = 0;
    lastBatchTime = 0;
    isInitialized = false;
    constructor(options) {
        if (options.ewmaAlpha < 0 || options.ewmaAlpha > 1) {
            throw new Error("ewmaAlpha must be between 0 and 1");
        }
        if (options.batchWindowMs <= 0) {
            throw new Error("batchWindowMs must be positive");
        }
        this.ewmaAlpha = options.ewmaAlpha;
        this.batchWindowMs = options.batchWindowMs;
    }
    /**
     * Adds a sample to the current batch
     */
    addSample(value, timestamp = Date.now()) {
        if (value < 0) {
            throw new Error("Queue length cannot be negative");
        }
        this.samples.push(value);
        // Update last batch time on first sample
        if (this.samples.length === 1) {
            this.lastBatchTime = timestamp;
        }
    }
    /**
     * Checks if enough time has passed to process the current batch
     */
    shouldProcessBatch(currentTime = Date.now()) {
        if (this.samples.length === 0) {
            return false;
        }
        return currentTime - this.lastBatchTime >= this.batchWindowMs;
    }
    calculateMedian(samples) {
        const sortedSamples = [...samples].sort((a, b) => a - b);
        const mid = Math.floor(sortedSamples.length / 2);
        if (sortedSamples.length % 2 === 1) {
            // Odd length: use middle value
            const median = sortedSamples[mid];
            if (median === undefined) {
                this.logger.error("Invalid median calculated from odd samples", {
                    sortedSamples,
                    mid,
                    median,
                });
                return null;
            }
            return median;
        }
        else {
            // Even length: average two middle values
            const lowMid = sortedSamples[mid - 1];
            const highMid = sortedSamples[mid];
            if (lowMid === undefined || highMid === undefined) {
                this.logger.error("Invalid median calculated from even samples", {
                    sortedSamples,
                    mid,
                    lowMid,
                    highMid,
                });
                return null;
            }
            const median = (lowMid + highMid) / 2;
            return median;
        }
    }
    /**
     * Processes the current batch of samples and returns the result.
     * Clears the samples array and updates the smoothed value.
     *
     * Returns null if there are no samples to process.
     */
    processBatch(currentTime = Date.now()) {
        if (this.samples.length === 0) {
            // No samples to process
            return null;
        }
        // Calculate median of samples to filter outliers
        const median = this.calculateMedian(this.samples);
        if (median === null) {
            // We already logged a more specific error message
            return null;
        }
        // Update EWMA smoothed value
        if (!this.isInitialized) {
            // First value - initialize with median
            this.smoothedValue = median;
            this.isInitialized = true;
        }
        else {
            // Apply EWMA: s_t = α * x_t + (1 - α) * s_(t-1)
            this.smoothedValue = this.ewmaAlpha * median + (1 - this.ewmaAlpha) * this.smoothedValue;
        }
        const result = {
            median,
            smoothedValue: this.smoothedValue,
            sampleCount: this.samples.length,
            samples: Object.freeze([...this.samples]),
        };
        // Clear samples for next batch
        this.samples = [];
        this.lastBatchTime = currentTime;
        return result;
    }
    /**
     * Gets the current smoothed value without processing a batch
     */
    getSmoothedValue() {
        return this.smoothedValue;
    }
    /**
     * Gets the number of samples in the current batch
     */
    getCurrentSampleCount() {
        return this.samples.length;
    }
    /**
     * Gets the current samples (for testing/debugging)
     */
    getCurrentSamples() {
        return Object.freeze([...this.samples]);
    }
    /**
     * Resets the processor state
     */
    reset() {
        this.samples = [];
        this.smoothedValue = 0;
        this.lastBatchTime = 0;
        this.isInitialized = false;
    }
    /**
     * Gets processor configuration
     */
    getConfig() {
        return {
            ewmaAlpha: this.ewmaAlpha,
            batchWindowMs: this.batchWindowMs,
        };
    }
}
//# sourceMappingURL=queueMetricsProcessor.js.map