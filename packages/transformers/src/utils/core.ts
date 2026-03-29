/**
 * @file Core utility functions/classes for Transformers.js.
 *
 * These are only used internally, meaning an end-user shouldn't
 * need to access anything here.
 *
 * @module utils/core
 */

/**
 * @typedef {Object} InitiateProgressInfo
 * @property {'initiate'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 */

/**
 * @typedef {Object} DownloadProgressInfo
 * @property {'download'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 */

/**
 * @typedef {Object} ProgressStatusInfo
 * @property {'progress'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 * @property {number} progress A number between 0 and 100.
 * @property {number} loaded The number of bytes loaded.
 * @property {number} total The total number of bytes to be loaded.
 */

/**
 * @typedef {Object} FileLoadingProgress
 * @property {number} loaded The number of bytes loaded for this file.
 * @property {number} total The total number of bytes for this file.
 */

/**
 * @typedef {Record<string, FileLoadingProgress>} FilesLoadingMap
 * A mapping of file names to their loading progress. Each key is a file path and each value contains
 * the loaded and total bytes for that file.
 */

/**
 * @typedef {Object} TotalProgressInfo
 * @property {'progress_total'} status
 * @property {string} name The model id or directory path.
 * @property {number} progress A number between 0 and 100.
 * @property {number} loaded The number of bytes loaded.
 * @property {number} total The total number of bytes to be loaded.
 * @property {FilesLoadingMap} files A mapping of file names to their loading progress.
 */

/**
 * @typedef {Object} DoneProgressInfo
 * @property {'done'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 */

/**
 * @typedef {Object} ReadyProgressInfo
 * @property {'ready'} status
 * @property {string} task The loaded task.
 * @property {string} model The loaded model.
 */

/**
 * @typedef {InitiateProgressInfo | DownloadProgressInfo | ProgressStatusInfo | DoneProgressInfo | ReadyProgressInfo | TotalProgressInfo} ProgressInfo
 */

export type ProgressInfo = InitiateProgressInfo | DownloadProgressInfo | ProgressStatusInfo | DoneProgressInfo | ReadyProgressInfo | TotalProgressInfo;

type InitiateProgressInfo = { status: 'initiate'; name: string; file: string };
type DownloadProgressInfo = { status: 'download'; name: string; file: string };
type ProgressStatusInfo = { status: 'progress'; name: string; file: string; progress: number; loaded: number; total: number };
type FileLoadingProgress = { loaded: number; total: number };
type TotalProgressInfo = { status: 'progress_total'; name: string; progress: number; loaded: number; total: number; files: Record<string, FileLoadingProgress> };
type DoneProgressInfo = { status: 'done'; name: string; file: string };
type ReadyProgressInfo = { status: 'ready'; task: string; model: string };

/**
 * A callback function that is called with progress information.
 * @callback ProgressCallback
 * @param {ProgressInfo} progressInfo
 * @returns {void}
 */
export type ProgressCallback = (progressInfo: ProgressInfo) => void;

/**
 * Helper function to dispatch progress callbacks.
 *
 * @param {ProgressCallback | null | undefined} progress_callback The progress callback function to dispatch.
 * @param {ProgressInfo} data The data to pass to the progress callback function.
 * @returns {void}
 * @private
 */
export function dispatchCallback(progress_callback: ProgressCallback | null | undefined, data: ProgressInfo): void {
    if (progress_callback) progress_callback(data);
}

/**
 * Reverses the keys and values of an object.
 *
 * @param {Object} data The object to reverse.
 * @returns {Object} The reversed object.
 * @see https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
 */
export function reverseDictionary(data: Record<string, unknown>): Record<string, unknown> {
    // https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
    return Object.fromEntries(Object.entries(data).map(([key, value]) => [value, key]));
}

/**
 * Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
 *
 * @param {string} string The string to escape.
 * @returns {string} The escaped string.
 */
export function escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
 * Check if a value is a typed array.
 * @param {*} val The value to check.
 * @returns {boolean} True if the value is a `TypedArray`, false otherwise.
 *
 * Adapted from https://stackoverflow.com/a/71091338/13989043
 */
export function isTypedArray(val: unknown): boolean {
    return (val as { prototype?: { __proto__?: { constructor?: { name?: string } } } })?.prototype?.__proto__?.constructor?.name === 'TypedArray';
}

/**
 * Check if a value is an integer.
 * @param {*} x The value to check.
 * @returns {boolean} True if the value is a string, false otherwise.
 */
export function isIntegralNumber(x: unknown): boolean {
    return Number.isInteger(x) || typeof x === 'bigint';
}

/**
 * Determine if a provided width or height is nullish.
 * @param {*} x The value to check.
 * @returns {boolean} True if the value is `null`, `undefined` or `-1`, false otherwise.
 */
export function isNullishDimension(x: unknown): boolean {
    return x === null || x === undefined || x === -1;
}

/**
 * Calculates the dimensions of a nested array.
 *
 * @param {any[]} arr The nested array to calculate dimensions for.
 * @returns {number[]} An array containing the dimensions of the input array.
 */
export function calculateDimensions(arr: unknown[]): number[] {
    const dimensions: number[] = [];
    let current: unknown = arr;
    while (Array.isArray(current)) {
        dimensions.push(current.length);
        current = current[0];
    }
    return dimensions;
}

/**
 * Replicate python's .pop() method for objects.
 * @param {Object} obj The object to pop from.
 * @param {string} key The key to pop.
 * @param {*} defaultValue The default value to return if the key does not exist.
 * @returns {*} The value of the popped key.
 * @throws {Error} If the key does not exist and no default value is provided.
 */
export function pop(obj: Record<string, unknown>, key: string, defaultValue: unknown = undefined): unknown {
    const value = obj[key];
    if (value !== undefined) {
        delete obj[key];
        return value;
    }
    if (defaultValue === undefined) {
        throw Error(`Key ${key} does not exist in object.`);
    }
    return defaultValue;
}

/**
 * Efficiently merge arrays, creating a new copy.
 * Adapted from https://stackoverflow.com/a/6768642/13989043
 * @param  {any[]} arrs Arrays to merge.
 * @returns {any[]} The merged array.
 */
export function mergeArrays(...arrs: unknown[][]): unknown[] {
    return Array.prototype.concat.apply([], arrs);
}

/**
 * Compute the Cartesian product of given arrays
 * @param {...any[]} a Arrays to compute the product
 * @returns {any[]} Returns the computed Cartesian product as an array
 * @private
 */
export function product(...a: unknown[][]): unknown[] {
    // Cartesian product of items
    // Adapted from https://stackoverflow.com/a/43053803
    return a.reduce((a: unknown[], b: unknown[]) => a.flatMap((d: unknown) => b.map((e: unknown) => [d, e])));
}

/**
 * Calculates the index offset for a given index and window size.
 * @param {number} i The index.
 * @param {number} w The window size.
 * @returns {number} The index offset.
 */
export function calculateReflectOffset(i: number, w: number): number {
    return Math.abs(((i + w) % (2 * w)) - w);
}

/**
 *
 * @param {Object} o
 * @param {string[]} props
 * @returns {Object}
 */
export function pick(o: Record<string, unknown>, props: string[]): Record<string, unknown> {
    return Object.assign(
        {},
        ...props.map((prop: string) => {
            if (o[prop] !== undefined) {
                return { [prop]: o[prop] };
            }
        }),
    );
}

/**
 * Calculate the length of a string, taking multi-byte characters into account.
 * This mimics the behavior of Python's `len` function.
 * @param {string} s The string to calculate the length of.
 * @returns {number} The length of the string.
 */
export function len(s: string): number {
    let length = 0;
    for (const c of s) ++length;
    return length;
}

/**
 * Count the occurrences of a value in an array or string.
 * This mimics the behavior of Python's `count` method.
 * @param {any[]|string} arr The array or string to search.
 * @param {any} value The value to count.
 */
export function count(arr: unknown[] | string, value: unknown): number {
    let count = 0;
    for (const v of arr) {
        if (v === value) ++count;
    }
    return count;
}
