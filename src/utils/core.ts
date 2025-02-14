/**
 * @file Core utility functions/classes for Transformers.js.
 *
 * These are only used internally, meaning an end-user shouldn't
 * need to access anything here.
 *
 * @module utils/core
 */

export interface InitiateProgressInfo {
    status: 'initiate';
    name: string; // The model id or directory path
    file: string; // The name of the file
}

export interface DownloadProgressInfo {
    status: 'download';
    name: string; // The model id or directory path
    file: string; // The name of the file
}

export interface ProgressStatusInfo {
    status: 'progress';
    name: string; // The model id or directory path
    file: string; // The name of the file
    progress: number; // A number between 0 and 100
    loaded: number; // The number of bytes loaded
    total: number; // The total number of bytes to be loaded
}

export interface DoneProgressInfo {
    status: 'done';
    name: string; // The model id or directory path
    file: string; // The name of the file
}

export interface ReadyProgressInfo {
    status: 'ready';
    task: string; // The loaded task
    model: string; // The loaded model
}

export type ProgressInfo = InitiateProgressInfo | DownloadProgressInfo | ProgressStatusInfo | DoneProgressInfo | ReadyProgressInfo;

export type ProgressCallback = (progressInfo: ProgressInfo) => void;

/**
 * Helper function to dispatch progress callbacks.
 */
export function dispatchCallback(progress_callback: ProgressCallback | null | undefined, data: ProgressInfo): void {
    if (progress_callback) progress_callback(data);
}

/**
 * Reverses the keys and values of an object.
 * @see https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
 */
export function reverseDictionary<T extends Record<string, string>>(data: T): Record<string, string> {
    return Object.fromEntries(Object.entries(data).map(([key, value]) => [value, key]));
}

/**
 * Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
 * @private
 */
export function escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
 * Check if a value is a typed array.
 * Adapted from https://stackoverflow.com/a/71091338/13989043
 */
export function isTypedArray(val: unknown): boolean {
    // return val?.prototype?.__proto__?.constructor?.name === 'TypedArray';
    return ArrayBuffer.isView(val) && !(val instanceof DataView);
}

/**
 * Check if a value is an integer.
 */
export function isIntegralNumber(x: unknown): boolean {
    return Number.isInteger(x as number) || typeof x === 'bigint';
}

/**
 * Determine if a provided width or height is nullish.
 */
export function isNullishDimension(x: unknown): boolean {
    return x === null || x === undefined || x === -1;
}

/**
 * Calculates the dimensions of a nested array.
 */
export function calculateDimensions(arr: any[]): number[] {
    const dimensions: number[] = [];
    let current: any = arr;
    while (Array.isArray(current)) {
        dimensions.push(current.length);
        current = current[0];
    }
    return dimensions;
}

/**
 * Replicate python's .pop() method for objects.
 * @throws {Error} If the key does not exist and no default value is provided.
 */
export function pop<T extends Record<string, any>, K extends keyof T>(
    obj: T,
    key: K,
    defaultValue?: T[K]
): T[K] {
    const value = obj[key];
    if (value !== undefined) {
        delete obj[key];
        return value;
    }
    if (defaultValue === undefined) {
        throw Error(`Key ${String(key)} does not exist in object.`);
    }
    return defaultValue;
}

/**
 * Efficiently merge arrays, creating a new copy.
 * Adapted from https://stackoverflow.com/a/6768642/13989043
 */
export function mergeArrays<T>(...arrs: T[][]): T[] {
    return Array.prototype.concat.apply([], arrs);
}

/**
 * Compute the Cartesian product of given arrays
 * @private
 */
export function product<T>(...a: T[][]): T[][] {
    // Cartesian product of items
    // Adapted from https://stackoverflow.com/a/43053803
    return a.reduce((acc, b) => acc.flatMap(d => b.map(e => [...d, e])), [[]] as T[][]);
}

/**
 * Calculates the index offset for a given index and window size.
 */
export function calculateReflectOffset(i: number, w: number): number {
    return Math.abs((i + w) % (2 * w) - w);
}

/**
 * Save blob file on the web.
 */
export function saveBlob(path: string, blob: Blob): void {
    // Convert the canvas content to a data URL
    const dataURL = URL.createObjectURL(blob);

    // Create an anchor element with the data URL as the href attribute
    const downloadLink = document.createElement('a');
    downloadLink.href = dataURL;

    // Set the download attribute to specify the desired filename for the downloaded image
    downloadLink.download = path;

    // Trigger the download
    downloadLink.click();

    // Clean up: remove the anchor element from the DOM
    downloadLink.remove();

    // Revoke the Object URL to free up memory
    URL.revokeObjectURL(dataURL);
}

/**
 * Pick specific properties from an object
 */
export function pick<T extends Record<string, any>, K extends keyof T>(
    o: T,
    props: K[]
): Pick<T, K> {
    return Object.assign(
        {},
        ...props.map((prop) => {
            if (o[prop] !== undefined) {
                return { [prop]: o[prop] };
            }
            return {};
        })
    );
}

/**
 * Calculate the length of a string, taking multi-byte characters into account.
 * This mimics the behavior of Python's `len` function.
 */
export function len(s: string): number {
    let length = 0;
    for (const c of s) ++length;
    return length;
}

/**
 * Count the occurrences of a value in an array or string.
 * This mimics the behavior of Python's `count` method.
 */
export function count<T>(arr: T[] | string, value: T): number {
    let count = 0;
    for (const v of arr) {
        if (v === value) ++count;
    }
    return count;
} 