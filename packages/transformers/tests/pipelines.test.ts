import { init } from "./init.ts";
import { collect_and_execute_pipeline_tests } from "./test_utils.ts";

// Initialise the testing environment
init();
await collect_and_execute_pipeline_tests("Pipelines");
