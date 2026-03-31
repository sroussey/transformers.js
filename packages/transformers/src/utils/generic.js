/**
 * A base class for creating callable objects.
 * See [here](https://stackoverflow.com/q/76073890) for more information.
 *
 * Instances of subclasses can be called directly as functions, e.g. `model(inputs)`.
 * The call is delegated to the `_call` method which subclasses must implement.
 */

/**
 * @typedef {((...args: any[]) => any) & { _call(...args: any[]): any }} CallableInstance
 */

/**
 * @typedef {new (...args: any[]) => CallableInstance} CallableConstructor
 */

export const Callable = /** @type {CallableConstructor} */ (/** @type {unknown} */ (class {
    constructor() {
        /** @param {any[]} args */
        let closure = function (...args) {
            return /** @type {CallableInstance} */ (/** @type {unknown} */ (closure))._call(...args);
        };
        return Object.setPrototypeOf(closure, new.target.prototype);
    }

    /** @param {any[]} _args */
    _call(..._args) {
        throw Error('Must implement _call method in subclass');
    }
}));
