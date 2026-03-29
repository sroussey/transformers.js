import { init } from "./init.ts";
import { collect_and_execute_tests } from "./test_utils.ts";

init();
await collect_and_execute_tests("Feature extractors", "feature_extraction");
