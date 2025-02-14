/**
 * @file Custom data structures.
 * 
 * These are only used internally, meaning an end-user shouldn't
 * need to access anything here.
 * 
 * @module utils/data-structures
 */

/**
 * Efficient Heap-based Implementation of a Priority Queue.
 * It uses an array-based binary heap, where the root is at index `0`, and the
 * children of node `i` are located at indices `2i + 1` and `2i + 2`, respectively.
 * 
 * Adapted from the following sources:
 * - https://stackoverflow.com/a/42919752/13989043 (original)
 * - https://github.com/belladoreai/llama-tokenizer-js (minor improvements)
 */
export class PriorityQueue<T> {
    private _heap: T[];
    private _comparator: (a: T, b: T) => boolean;
    private _maxSize: number;

    /**
     * Create a new PriorityQueue.
     * @param {function(T, T): boolean} comparator Comparator function to determine priority. Defaults to a MaxHeap.
     */
    constructor(comparator: (a: T, b: T) => boolean = (a, b) => a > b, maxSize: number = Infinity) {
        this._heap = [];
        this._comparator = comparator;
        this._maxSize = maxSize;
    }

    /**
     * The size of the queue
     */
    get size(): number {
        return this._heap.length;
    }

    /**
     * Check if the queue is empty.
     */
    isEmpty(): boolean {
        return this.size === 0;
    }

    /**
     * Return the element with the highest priority in the queue.
     */
    peek(): T | undefined {
        return this._heap[0];
    }

    /**
     * Add one or more elements to the queue.
     */
    push(...values: T[]): number {
        return this.extend(values);
    }

    /**
     * Add multiple elements to the queue.
     */
    extend(values: T[]): number {
        for (const value of values) {
            if (this.size < this._maxSize) {
                this._heap.push(value);
                this._siftUp();
            } else {
                // Get index of value with the lowest priority
                const smallest = this._smallest();

                // If the new value has higher priority than the smallest value in the heap
                // then replace the smallest value with the new value and update the heap
                if (this._comparator(value, this._heap[smallest])) {
                    this._heap[smallest] = value;
                    this._siftUpFrom(smallest);
                }
            }
        }
        return this.size;
    }

    /**
     * Remove and return the element with the highest priority in the queue.
     */
    pop(): T | undefined {
        const poppedValue = this.peek();
        const bottom = this.size - 1;
        if (bottom > 0) {
            this._swap(0, bottom);
        }
        this._heap.pop();
        this._siftDown();
        return poppedValue;
    }

    /**
     * Replace the element with the highest priority in the queue with a new value.
     */
    replace(value: T): T | undefined {
        const replacedValue = this.peek();
        this._heap[0] = value;
        this._siftDown();
        return replacedValue;
    }

    /**
     * Get the index of the parent of the node at index `i`.
     */
    private _parent(i: number): number {
        return ((i + 1) >>> 1) - 1;
    }

    /**
     * Get the index of the left child of the node at index `i`.
     */
    private _left(i: number): number {
        return (i << 1) + 1;
    }

    /**
     * Get the index of the right child of the node at index `i`.
     */
    private _right(i: number): number {
        return (i + 1) << 1;
    }

    /**
     * Check if the element at index `i` is greater than the element at index `j`.`true` if the element at index `i` is greater than the element at index `j`, `false` otherwise.
     * @private
     */
    private _greater(i: number, j: number): boolean {
        return this._comparator(this._heap[i], this._heap[j]);
    }

    /**
     * Swap the elements at indices `i` and `j`.
     */
    private _swap(i: number, j: number): void {
        [this._heap[i], this._heap[j]] = [this._heap[j], this._heap[i]];
    }

    /**
     * Maintain the heap property by updating positions in the heap,
     * starting at the last element and moving up the heap.
     */
    private _siftUp(): void {
        this._siftUpFrom(this.size - 1);
    }

   /**
     * Helper function to sift up from a given node.
     */
   private _siftUpFrom(node: number): void {
        while (node > 0 && this._greater(node, this._parent(node))) {
            this._swap(node, this._parent(node));
            node = this._parent(node);
        }
    }

    /**
     * Maintain the heap property by updating positions in the heap,
     * starting at the first element and moving down the heap.
     */
    private _siftDown(): void {
        let node = 0;
        while (
            (this._left(node) < this.size && this._greater(this._left(node), node)) ||
            (this._right(node) < this.size && this._greater(this._right(node), node))
        ) {
            const maxChild = (this._right(node) < this.size && this._greater(this._right(node), this._left(node)))
                ? this._right(node)
                : this._left(node);
            this._swap(node, maxChild);
            node = maxChild;
        }
    }
    /**
     * Get the index of the smallest element in the heap. Since we use an array-based heap,
     * the index can be computed without needing to traverse the heap.
     */
    private _smallest(): number {
        return (2 ** (Math.floor(Math.log2(this.size))) - 1);
    }
}

/**
 * A trie structure to efficiently store and search for strings.
 */
export class CharTrie {
    root: CharTrieNode;

    constructor() {
        this.root = CharTrieNode.default();
    }

    /**
     * Adds one or more `texts` to the trie.
     */
    extend(texts: string[]): void {
        for (const text of texts) {
            this.push(text);
        }
    }

    /**
     * Adds text to the trie.
     */
    push(text: string): void {
        let node = this.root;
        for (const ch of text) {
            let child = node.children.get(ch);
            if (child === undefined) {
                child = CharTrieNode.default();
                node.children.set(ch, child);
            }
            node = child;
        }
        node.isLeaf = true;
    }

    /**
     * Searches the trie for all strings with a common prefix of `text`.
     * Yields each string in the trie that has `text` as a prefix.
     */
    *commonPrefixSearch(text: string): Generator<string> {
        let node = this.root;
        if (node === undefined) return;

        let prefix = "";
        for (const ch of text) {
            prefix += ch;
            node = node.children.get(ch);
            if (node === undefined) return;
            if (node.isLeaf) {
                yield prefix;
            }
        }
    }
}

/**
 * Represents a node in a character trie.
 */
class CharTrieNode {
    isLeaf: boolean;
    children: Map<string, CharTrieNode>;

    /**
     * Create a new CharTrieNode.
     */
    constructor(isLeaf: boolean, children: Map<string, CharTrieNode>) {
        this.isLeaf = isLeaf;
        this.children = children;
    }

    /**
     * Returns a new `CharTrieNode` instance with default values.
     */
    static default(): CharTrieNode {
        return new CharTrieNode(false, new Map());
    }
}

/**
 * A lattice data structure to be used for tokenization.
 */
export class TokenLattice {
    private chars: string[];
    private len: number;
    private bosTokenId: number;
    private eosTokenId: number;
    private nodes: TokenLatticeNode[];
    private beginNodes: TokenLatticeNode[][];
    private endNodes: TokenLatticeNode[][];

    /**
     * Creates a new TokenLattice instance.
     */
    constructor(sentence: string, bosTokenId: number, eosTokenId: number) {
        this.chars = Array.from(sentence);
        this.len = this.chars.length;
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.nodes = [];
        this.beginNodes = Array.from({ length: this.len + 1 }, () => []);
        this.endNodes = Array.from({ length: this.len + 1 }, () => []);

        const bos = new TokenLatticeNode(this.bosTokenId, 0, 0, 0, 0.0);
        const eos = new TokenLatticeNode(this.eosTokenId, 1, this.len, 0, 0.0);
        this.nodes.push(bos.clone());
        this.nodes.push(eos.clone());
        this.beginNodes[this.len].push(eos);
        this.endNodes[0].push(bos);
    }

    /**
     * Inserts a new token node into the token lattice.
     */
    insert(pos: number, length: number, score: number, tokenId: number): void {
        const nodeId = this.nodes.length;
        const node = new TokenLatticeNode(tokenId, nodeId, pos, length, score);
        this.beginNodes[pos].push(node);
        this.endNodes[pos + length].push(node);
        this.nodes.push(node);
    }

    /**
     * Implements the Viterbi algorithm to compute the most likely sequence of tokens.
     */
    viterbi(): TokenLatticeNode[] {
        const len = this.len;
        let pos = 0;
        while (pos <= len) {
            if (this.beginNodes[pos].length == 0) {
                return [];
            }
            for (let rnode of this.beginNodes[pos]) {
                rnode.prev = null;
                let bestScore = 0.0;
                let bestNode: TokenLatticeNode | null = null;
                for (let lnode of this.endNodes[pos]) {
                    const score = lnode.backtraceScore + rnode.score;
                    if (bestNode === null || score > bestScore) {
                        bestNode = lnode.clone();
                        bestScore = score;
                    }
                }

                if (bestNode !== null) {
                    rnode.prev = bestNode;
                    rnode.backtraceScore = bestScore;
                } else {
                    return [];
                }
            }
            ++pos;
        }

        const results: TokenLatticeNode[] = [];
        const root = this.beginNodes[len][0];
        const prev = root.prev;
        if (prev === null) {
            return [];
        }

        let node = prev.clone();
        while (node.prev !== null) {
            results.push(node.clone());
            const n = node.clone();
            node = n.prev.clone();
        }

        results.reverse();
        return results;
    }

    /**
     * Returns the piece of the sentence that corresponds to the given token node.
     */
    piece(node: TokenLatticeNode): string {
        return this.chars.slice(node.pos, node.pos + node.length).join('');
    }

    /**
     * Returns the most likely sequence of tokens.
     */
    tokens(): string[] {
        const nodes = this.viterbi();
        return nodes.map(x => this.piece(x));
    }

    /**
     * Returns the most likely sequence of token IDs.
     */
    tokenIds(): number[] {
        const nodes = this.viterbi();
        return nodes.map(x => x.tokenId);
    }
}

class TokenLatticeNode {
    tokenId: number;
    nodeId: number;
    pos: number;
    length: number;
    score: number;
    prev: TokenLatticeNode | null;
    backtraceScore: number;

    /**
     * Represents a node in a token lattice for a given sentence.
     * @param {number} tokenId The ID of the token associated with this node.
     * @param {number} nodeId The ID of this node.
     * @param {number} pos The starting position of the token in the sentence.
     * @param {number} length The length of the token.
     * @param {number} score The score associated with the token.
     */
    constructor(tokenId: number, nodeId: number, pos: number, length: number, score: number) {
        this.tokenId = tokenId;
        this.nodeId = nodeId;
        this.pos = pos;
        this.length = length;
        this.score = score;
        this.prev = null;
        this.backtraceScore = 0.0;
    }

    /**
     * Returns a clone of this node.
     */
    clone(): TokenLatticeNode {
        const n = new TokenLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score);
        n.prev = this.prev;
        n.backtraceScore = this.backtraceScore;
        return n;
    }
} 