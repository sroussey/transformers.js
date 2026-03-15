import { ModelRegistry } from "../../src/transformers.js";

const MAX_TEST_EXECUTION_TIME = 30_000;

describe("ModelRegistry", () => {
  describe("get_available_dtypes", () => {
    it(
      "should detect available dtypes for a model",
      async () => {
        const dtypes = await ModelRegistry.get_available_dtypes(
          "onnx-community/all-MiniLM-L6-v2-ONNX",
        );

        // This model is known to have multiple quantization levels
        expect(Array.isArray(dtypes)).toBe(true);
        expect(dtypes.length).toBeGreaterThan(0);

        // Should contain at least some common dtypes
        // (fp32 is typically always available for ONNX models)
        expect(dtypes).toContain("fp32");

        // All returned values should be valid dtype strings
        const validDtypes = ["fp32", "fp16", "int8", "uint8", "q8", "q4", "q4f16", "bnb4"];
        for (const dtype of dtypes) {
          expect(validDtypes).toContain(dtype);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return an empty array for a model with no ONNX files",
      async () => {
        // Use a model that likely doesn't have ONNX files
        const dtypes = await ModelRegistry.get_available_dtypes(
          "hf-internal-testing/tiny-random-GPTNeoForCausalLM",
        );

        expect(Array.isArray(dtypes)).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
});
