export interface QueueMetricsProcessorOptions {
    /**
     * EWMA smoothing factor (0-1)
     * Lower values = more smoothing, less reactive
     * Higher values = more responsive to recent changes
     */
    ewmaAlpha: number;
    /**
     * Batch window duration in milliseconds
     * Samples within this window are collected and processed together
     */
    batchWindowMs: number;
}
export interface BatchProcessingResult {
    /** Median of samples in the batch */
    median: number;
    /** EWMA-smoothed value after processing this batch */
    smoothedValue: number;
    /** Number of samples processed in this batch */
    sampleCount: number;
    /** Raw samples that were processed */
    samples: readonly number[];
}
/**
 * Processes queue length samples using exponential weighted moving average (EWMA)
 * for smoothing and median filtering for outlier resistance.
 *
 * Collects samples within a batch window, calculates median to filter outliers,
 * then applies EWMA smoothing for stable trend tracking.
 */
export declare class QueueMetricsProcessor {
    private readonly ewmaAlpha;
    private readonly batchWindowMs;
    private readonly logger;
    private samples;
    private smoothedValue;
    private lastBatchTime;
    private isInitialized;
    constructor(options: QueueMetricsProcessorOptions);
    /**
     * Adds a sample to the current batch
     */
    addSample(value: number, timestamp?: number): void;
    /**
     * Checks if enough time has passed to process the current batch
     */
    shouldProcessBatch(currentTime?: number): boolean;
    private calculateMedian;
    /**
     * Processes the current batch of samples and returns the result.
     * Clears the samples array and updates the smoothed value.
     *
     * Returns null if there are no samples to process.
     */
    processBatch(currentTime?: number): BatchProcessingResult | null;
    /**
     * Gets the current smoothed value without processing a batch
     */
    getSmoothedValue(): number;
    /**
     * Gets the number of samples in the current batch
     */
    getCurrentSampleCount(): number;
    /**
     * Gets the current samples (for testing/debugging)
     */
    getCurrentSamples(): readonly number[];
    /**
     * Resets the processor state
     */
    reset(): void;
    /**
     * Gets processor configuration
     */
    getConfig(): Readonly<QueueMetricsProcessorOptions>;
}
