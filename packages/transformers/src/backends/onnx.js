/**
 * @file Handler file for choosing the correct version of ONNX Runtime, based on the environment.
 * Ideally, we could import the `onnxruntime-web` and `onnxruntime-node` packages only when needed,
 * but dynamic imports don't seem to work with the current webpack version and/or configuration.
 * This is possibly due to the experimental nature of top-level await statements.
 * So, we just import both packages, and use the appropriate one based on the environment:
 *   - When running in node, we use `onnxruntime-node`.
 *   - When running in the browser, we use `onnxruntime-web` (`onnxruntime-node` is not bundled).
 *
 * This module is not directly exported, but can be accessed through the environment variables:
 * ```javascript
 * import { env } from '@huggingface/transformers';
 * console.log(env.backends.onnx);
 * ```
 *
 * @module backends/onnx
 */

import { apis, env, LogLevel } from '../env.js';

// NOTE: Import order matters here. We need to import `onnxruntime-node` before `onnxruntime-web`.
// In either case, we select the default export if it exists, otherwise we use the named export.
import * as ONNX_NODE from 'onnxruntime-node';
import * as ONNX_WEB from 'onnxruntime-web/webgpu';
import { isBlobURL, toAbsoluteURL } from '../utils/hub/utils.js';
import { logger } from '../utils/logger.js';
import { loadWasmBinary, loadWasmFactory } from './utils/cacheWasm.js';
export { Tensor } from 'onnxruntime-common';

/**
 * @typedef {import('onnxruntime-common').InferenceSession.ExecutionProviderConfig} ONNXExecutionProviders
 */

/** @type {Record<import("../utils/devices.js").DeviceType, ONNXExecutionProviders>} */
const DEVICE_TO_EXECUTION_PROVIDER_MAPPING = Object.freeze({
    auto: null, // Auto-detect based on device and environment
    gpu: null, // Auto-detect GPU
    cpu: 'cpu', // CPU
    wasm: 'wasm', // WebAssembly
    webgpu: 'webgpu', // WebGPU
    cuda: 'cuda', // CUDA
    dml: 'dml', // DirectML
    coreml: 'coreml', // CoreML

    webnn: { name: 'webnn', deviceType: 'cpu' }, // WebNN (default)
    'webnn-npu': { name: 'webnn', deviceType: 'npu' }, // WebNN NPU
    'webnn-gpu': { name: 'webnn', deviceType: 'gpu' }, // WebNN GPU
    'webnn-cpu': { name: 'webnn', deviceType: 'cpu' }, // WebNN CPU
});

/**
 * Converts any LogLevel value to ONNX Runtime's numeric severity level (0-4).
 * This handles both standard LogLevel values (10, 20, 30, 40, 50) and custom intermediate values.
 *
 * @param {number} logLevel - The LogLevel value to convert
 * @returns {0 | 1 | 2 | 3 | 4} ONNX Runtime severity level (0-4)
 */
function getOnnxLogSeverityLevel(logLevel) {
    if (logLevel <= LogLevel.DEBUG) {
        return 0; // ORT_LOGGING_LEVEL_VERBOSE
    } else if (logLevel <= LogLevel.INFO) {
        return 2; // ORT_LOGGING_LEVEL_WARNING
    } else if (logLevel <= LogLevel.WARNING) {
        return 3; // ORT_LOGGING_LEVEL_ERROR
    } else if (logLevel <= LogLevel.ERROR) {
        return 3; // ORT_LOGGING_LEVEL_ERROR
    } else {
        return 4; // ORT_LOGGING_LEVEL_FATAL
    }
}

/**
 * Maps ONNX Runtime numeric severity levels to string log levels.
 * @type {Record<0 | 1 | 2 | 3 | 4, 'verbose' | 'info' | 'warning' | 'error' | 'fatal'>}
 */
const ONNX_LOG_LEVEL_NAMES = {
    0: 'verbose',
    1: 'info',
    2: 'warning',
    3: 'error',
    4: 'fatal',
};

/**
 * The list of supported devices, sorted by priority/performance.
 * @type {import("../utils/devices.js").DeviceType[]}
 */
const supportedDevices = [];

/** @type {ONNXExecutionProviders[]} */
let defaultDevices;
/** @type {typeof ONNX_WEB} */
let ONNX;
const ORT_SYMBOL = Symbol.for('onnxruntime');

if (ORT_SYMBOL in globalThis) {
    // If the JS runtime exposes their own ONNX runtime, use it
    ONNX = /** @type {typeof ONNX_WEB} */ (/** @type {Record<symbol, unknown>} */ (globalThis)[ORT_SYMBOL]);
} else if (apis.IS_NODE_ENV) {
    ONNX = ONNX_NODE;

    // Updated as of ONNX Runtime 1.23.0-dev.20250612-70f14d7670
    switch (process.platform) {
        case 'win32': // Windows x64 and Windows arm64
            supportedDevices.push('dml');
            break;
        case 'linux': // Linux x64 and Linux arm64
            if (process.arch === 'x64') {
                supportedDevices.push('cuda');
            }
            break;
        case 'darwin': // MacOS x64 and MacOS arm64
            supportedDevices.push('coreml');
            break;
    }

    supportedDevices.push('webgpu');
    supportedDevices.push('cpu');
    defaultDevices = ['cpu'];
} else {
    ONNX = ONNX_WEB;

    if (apis.IS_WEBNN_AVAILABLE) {
        // TODO: Only push supported providers (depending on available hardware)
        supportedDevices.push('webnn-npu', 'webnn-gpu', 'webnn-cpu', 'webnn');
    }

    if (apis.IS_WEBGPU_AVAILABLE) {
        supportedDevices.push('webgpu');
    }

    supportedDevices.push('wasm');
    defaultDevices = ['wasm'];
}

/** @type {typeof import('onnxruntime-common').InferenceSession} */
const InferenceSession = /** @type {typeof ONNX_WEB} */ (ONNX).InferenceSession;

/**
 * Map a device to the execution providers to use for the given device.
 * @param {import("../utils/devices.js").DeviceType|"auto"|null} [device=null] (Optional) The device to run the inference on.
 * @returns {ONNXExecutionProviders[]} The execution providers to use for the given device.
 */
export function deviceToExecutionProviders(device = null) {
    // Use the default execution providers if the user hasn't specified anything
    if (!device) return defaultDevices;

    // Handle overloaded cases
    switch (device) {
        case 'auto':
            return supportedDevices;
        case 'gpu':
            return supportedDevices.filter((x) => ['webgpu', 'cuda', 'dml', 'webnn-gpu'].includes(x));
    }

    if (supportedDevices.includes(device)) {
        return [/** @type {Record<string, ONNXExecutionProviders>} */ (DEVICE_TO_EXECUTION_PROVIDER_MAPPING)[device] ?? device];
    }

    throw new Error(`Unsupported device: "${device}". Should be one of: ${supportedDevices.join(', ')}.`);
}

/**
 * Currently, Transformers.js doesn't support simultaneous loading of sessions in WASM/WebGPU.
 * For this reason, we need to chain the loading calls.
 * @type {Promise<any>}
 */
let webInitChain = Promise.resolve();

/**
 * Promise that resolves when WASM binary has been loaded (if caching is enabled).
 * This ensures we only attempt to load the WASM binary once.
 * @type {Promise<void>|null}
 */
let wasmLoadPromise = null;

/**
 * Ensures the WASM binary is loaded and cached before creating an inference session.
 * Only runs once, even if called multiple times.
 *
 * @returns {Promise<void>}
 */
async function ensureWasmLoaded() {
    // If already loading or loaded, return the existing promise
    if (wasmLoadPromise) {
        return wasmLoadPromise;
    }

    // Check if we should load the WASM binary
    const shouldUseWasmCache =
        env.useWasmCache &&
        typeof ONNX_ENV?.wasm?.wasmPaths === 'object' &&
        ONNX_ENV?.wasm?.wasmPaths?.wasm &&
        ONNX_ENV?.wasm?.wasmPaths?.mjs;

    if (!shouldUseWasmCache) {
        if (apis.IS_DENO_WEB_RUNTIME) {
            throw new Error(
                "env.useWasmCache=false is not supported in Deno's web runtime. Remove the useWasmCache override.",
            );
        }
        wasmLoadPromise = Promise.resolve();
        return wasmLoadPromise;
    }

    // Start loading the WASM binary
    wasmLoadPromise = (async () => {
        const urls = /** @type {{ wasm: string, mjs: string }} */ (/** @type {unknown} */ (ONNX_ENV.wasm.wasmPaths));

        let wasmBinaryLoaded = false;
        await Promise.all([
            // Load and cache the WASM binary
            urls.wasm && !isBlobURL(urls.wasm)
                ? (async () => {
                      try {
                          const wasmBinary = await loadWasmBinary(toAbsoluteURL(urls.wasm));
                          if (wasmBinary) {
                              ONNX_ENV.wasm.wasmBinary = wasmBinary;
                              wasmBinaryLoaded = true;
                          }
                      } catch (err) {
                          logger.warn('Failed to pre-load WASM binary:', err);
                      }
                  })()
                : Promise.resolve(),

            // Load and cache the WASM factory as a blob URL
            urls.mjs && !isBlobURL(urls.mjs)
                ? (async () => {
                      try {
                          const wasmFactoryBlob = await loadWasmFactory(toAbsoluteURL(urls.mjs));
                          if (wasmFactoryBlob) {
                              /** @type {Record<string, unknown>} */ (ONNX_ENV.wasm.wasmPaths).mjs = wasmFactoryBlob;
                          }
                      } catch (err) {
                          logger.warn('Failed to pre-load WASM factory:', err);
                      }
                  })()
                : Promise.resolve(),
        ]);

        // If wasmBinary failed to load, revert wasmPaths.mjs to the original URL
        if (!wasmBinaryLoaded) {
            /** @type {Record<string, unknown>} */ (ONNX_ENV.wasm.wasmPaths).mjs = urls.mjs;
        }
    })();

    return wasmLoadPromise;
}

/**
 * Create an ONNX inference session.
 * @param {Uint8Array|string} buffer_or_path The ONNX model buffer or path.
 * @param {import('onnxruntime-common').InferenceSession.SessionOptions} session_options ONNX inference session options.
 * @param {Object} session_config ONNX inference session configuration.
 * @returns {Promise<import('onnxruntime-common').InferenceSession & { config: Object }>} The ONNX inference session.
 */
export async function createInferenceSession(buffer_or_path, session_options, session_config) {
    await ensureWasmLoaded();
    const logSeverityLevel = getOnnxLogSeverityLevel(env.logLevel ?? LogLevel.WARNING);
    const load = () =>
        InferenceSession.create(/** @type {Uint8Array} */ (buffer_or_path), {
            // Set default log severity level, but allow overriding through session options
            logSeverityLevel,
            ...session_options,
        });
    const session = /** @type {import('onnxruntime-common').InferenceSession & { config: Record<string, unknown> }} */ (await (apis.IS_WEB_ENV ? (webInitChain = webInitChain.then(load)) : load()));
    session.config = /** @type {Record<string, unknown>} */ (session_config);
    return session;
}

/**
 * Currently, Transformers.js doesn't support simultaneous execution of sessions in WASM/WebGPU.
 * For this reason, we need to chain the inference calls (otherwise we get "Error: Session already started").
 * @type {Promise<any>}
 */
let webInferenceChain = Promise.resolve();

/**
 * Run an inference session.
 * @param {import('onnxruntime-common').InferenceSession} session The ONNX inference session.
 * @param {Record<string, import('onnxruntime-common').Tensor>} ortFeed The input tensors.
 * @returns {Promise<Record<string, import('onnxruntime-common').Tensor>>} The output tensors.
 */
export async function runInferenceSession(session, ortFeed) {
    const run = () => session.run(ortFeed);
    return apis.IS_WEB_ENV ? /** @type {Promise<Record<string, import('onnxruntime-common').Tensor>>} */ (webInferenceChain = webInferenceChain.then(run)) : run();
}

/**
 * Check if an object is an ONNX tensor.
 * @param {any} x The object to check
 * @returns {boolean} Whether the object is an ONNX tensor.
 */
export function isONNXTensor(x) {
    return x instanceof /** @type {typeof ONNX_WEB} */ (ONNX).Tensor;
}
/** @type {import('onnxruntime-common').Env} */
const ONNX_ENV = /** @type {typeof ONNX_WEB} */ (ONNX)?.env;

/**
 * Check if ONNX's WASM backend is being proxied.
 * @returns {boolean} Whether ONNX's WASM backend is being proxied.
 */
export function isONNXProxy() {
    // TODO: Update this when allowing non-WASM backends.
    return ONNX_ENV?.wasm?.proxy;
}

if (ONNX_ENV) {
    if (ONNX_ENV.wasm) {
        // Initialize wasm backend with suitable default settings.

        if (
            !(typeof /** @type {Record<string, unknown>} */ (globalThis).ServiceWorkerGlobalScope !== 'undefined' && self instanceof /** @type {{ new (): EventTarget }} */ (/** @type {Record<string, unknown>} */ (globalThis).ServiceWorkerGlobalScope)) &&
            ONNX_ENV.versions?.web &&
            !ONNX_ENV.wasm.wasmPaths
        ) {
            const wasmPathPrefix = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_ENV.versions.web}/dist/`;

            ONNX_ENV.wasm.wasmPaths = apis.IS_SAFARI
                ? {
                      mjs: `${wasmPathPrefix}ort-wasm-simd-threaded.mjs`,
                      wasm: `${wasmPathPrefix}ort-wasm-simd-threaded.wasm`,
                  }
                : {
                      mjs: `${wasmPathPrefix}ort-wasm-simd-threaded.asyncify.mjs`,
                      wasm: `${wasmPathPrefix}ort-wasm-simd-threaded.asyncify.wasm`,
                  };
        }

        ONNX_ENV.wasm.proxy = false;
    }

    if (ONNX_ENV.webgpu) {
        ONNX_ENV.webgpu.powerPreference = 'high-performance';
    }

    /**
     * A function to map Transformers.js log levels to ONNX Runtime log severity
     * levels, and set the log level environment variable in ONNX Runtime.
     * @param {number} logLevel The log level to set.
     */
    function setLogLevel(logLevel) {
        const severityLevel = /** @type {0 | 1 | 2 | 3 | 4} */ (getOnnxLogSeverityLevel(logLevel));
        ONNX_ENV.logLevel = /** @type {typeof ONNX_ENV.logLevel} */ (ONNX_LOG_LEVEL_NAMES[severityLevel]);
    }

    // Set the initial log level to be the default Transformers.js log level.
    setLogLevel(env.logLevel ?? LogLevel.WARNING);

    // Expose ONNX environment variables to `env.backends.onnx`
    env.backends.onnx = {
        ...ONNX_ENV,
        setLogLevel,
    };
}
