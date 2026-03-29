/**
 * A base class for creating callable objects.
 * See [here](https://stackoverflow.com/q/76073890) for more information.
 *
 * Instances of subclasses can be called directly as functions, e.g. `model(inputs)`.
 * The call is delegated to the `_call` method which subclasses must implement.
 */

interface CallableConstructor {
    new(): Callable;
}

export interface Callable {
    (...args: any[]): any;
    _call(...args: any[]): any;
}

export const Callable: CallableConstructor = class {
    constructor() {
        let closure = function (...args: unknown[]) {
            return (closure as unknown as Callable)._call(...args);
        };
        return Object.setPrototypeOf(closure, new.target.prototype);
    }

    _call(..._args: unknown[]): unknown {
        throw Error('Must implement _call method in subclass');
    }
} as unknown as CallableConstructor;
