import { DebertaV2Tokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, BERT_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = DebertaV2Tokenizer;
export const TEST_CONFIG = {
  "Xenova/nli-deberta-v3-small": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581doing", "?"],
      ids: [1, 577, 281, 274, 653, 302, 2],
      decoded: "[CLS] How are you doing?[SEP]",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["\u2581You", "\u2581should", "'", "ve", "\u2581done", "\u2581this"],
      ids: [1, 367, 403, 280, 415, 619, 291, 2],
      decoded: "[CLS] You should've done this[SEP]",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u25810", "123456", "789", "\u25810", "\u25811", "\u25812", "\u25813", "\u25814", "\u25815", "\u25816", "\u25817", "\u25818", "\u25819", "\u258110", "\u2581100", "\u25811000"],
      ids: [1, 767, 120304, 51535, 767, 376, 392, 404, 453, 456, 525, 574, 578, 712, 466, 803, 4985, 2],
      decoded: "[CLS] 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000[SEP]",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581founded", "\u2581in", "\u25812016", "."],
      ids: [1, 279, 483, 284, 3679, 267, 892, 260, 2],
      decoded: "[CLS] The company was founded in 2016.[SEP]",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "\u2581'", "ll", "\u2581!", "!", "to", "?", "'", "d", "'", "'", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [1, 336, 382, 436, 1084, 300, 725, 302, 280, 407, 280, 280, 407, 265, 261, 295, 280, 297, 260, 2],
      decoded: "[CLS] A 'll!!to?'d''d of, can't.[SEP]",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581def", "\u2581main", "(", ")", ":", "\u2581pass"],
      ids: [1, 23097, 872, 555, 285, 294, 1633, 2],
      decoded: "[CLS] def main(): pass[SEP]",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581a", "\u2581=", "\u2581obj", ".", "to", "String", "(", ")", ";", "\u2581to", "String", "(", ")", ";"],
      ids: [1, 678, 266, 1842, 68215, 260, 725, 29867, 555, 285, 346, 264, 29867, 555, 285, 346, 2],
      decoded: "[CLS] let a = obj.toString(); toString();[SEP]",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581This", "\u2581is", "\u2581a", "\u2581test", "."],
      ids: [1, 329, 269, 266, 1010, 260, 2],
      decoded: "[CLS] This is a test.[SEP]",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581UN", "want", "\u00e9", "d", ",", "running"],
      ids: [1, 4647, 27364, 5858, 407, 261, 15243, 2],
      decoded: "[CLS] UNwant\u00e9d,running[SEP]",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["\u25811", "\u0000", "2", "\u25813"],
      ids: [1, 376, 3, 445, 404, 2],
      decoded: "[CLS] 1[UNK]2 3[SEP]",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["\u2581Hello", "\u2581World"],
      ids: [1, 5365, 964, 2],
      decoded: "[CLS] Hello World[SEP]",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["\u2581hello", "\u2581world"],
      ids: [1, 12018, 447, 2],
      decoded: "[CLS] hello world[SEP]",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "\u751f", "\u6d3b", "\u7684", "\u771f", "\u8c1b", "\u662f"],
      ids: [1, 507, 41065, 101952, 9301, 98186, 3, 30060, 2],
      decoded: "[CLS] \u751f\u6d3b\u7684\u771f[UNK]\u662f[SEP]",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581leading", "\u2581space"],
      ids: [1, 1249, 754, 2],
      decoded: "[CLS] leading space[SEP]",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trailing", "\u2581space"],
      ids: [1, 18347, 754, 2],
      decoded: "[CLS] trailing space[SEP]",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["\u2581Hi", "\u2581Hello"],
      ids: [1, 2684, 5365, 2],
      decoded: "[CLS] Hi Hello[SEP]",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$", "1", "\u2581R", "2", "\u2581#", "3", "\u2581\u20ac4", "\u2581\u00a35", "\u2581\u00a5", "6", "\u2581", "\u20a3", "7", "\u2581\u20b9", "8", "\u2581\u20b1", "9", "\u2581test"],
      ids: [1, 1010, 419, 435, 909, 445, 953, 508, 56238, 14636, 56478, 765, 507, 3, 819, 34880, 804, 121499, 1088, 1010, 2],
      decoded: "[CLS] test $1 R2 #3 \u20ac4 \u00a35 \u00a56 [UNK]7 \u20b98 \u20b19 test[SEP]",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581bought", "\u2581an", "\u2581apple", "\u2581for", "\u2581$", "1", ".", "00", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [1, 273, 2031, 299, 6038, 270, 419, 435, 260, 962, 288, 262, 1106, 260, 2],
      decoded: "[CLS] I bought an apple for $1.00 at the store.[SEP]",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", ".", ".", "."],
      ids: [1, 274, 260, 260, 260, 2],
      decoded: "[CLS] you...[SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", ".", ".", "."],
      ids: [1, 274, 260, 260, 260, 2],
      decoded: "[CLS] you...[SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", ".", ".", ".", "\u2581you", ".", ".", "."],
      ids: [1, 274, 260, 260, 260, 274, 260, 260, 260, 2],
      decoded: "[CLS] you... you...[SEP]",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581weird", "\u2581", "\uff5e", "\u2581edge", "\u2581", "\uff5e", "\u2581case"],
      ids: [1, 4926, 507, 96622, 2363, 507, 96622, 571, 2],
      decoded: "[CLS] weird \uff5e edge \uff5e case[SEP]",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581This", "\u2581is", "\u2581a", "\u2581test", "\u2581."],
      ids: [1, 329, 269, 266, 1010, 323, 2],
      decoded: "[CLS] This is a test.[SEP]",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581\ud83d\ude02", "\u2581", "\ud83d\udc4d", "\u2581", "\ud83e\udd23", "\u2581", "\ud83d\ude0d", "\u2581", "\ud83d\ude2d", "\u2581", "\ud83c\udf89", "\u2581", "\ud83d\ude4f", "\u2581\ud83d\ude0a", "\u2581\ud83d\udd25", "\u2581", "\ud83d\ude01", "\u2581", "\ud83d\ude05", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\ude06", "\u2581", "\ud83d\udc4f", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc9c", "\u2581", "\ud83d\udc9a", "\u2581", "\ud83d\udc97", "\u2581", "\ud83d\udc99", "\u2581", "\ud83d\udda4", "\u2581", "\ud83d\ude0e", "\u2581", "\ud83d\udc4c", "\u2581", "\ud83e\udd73", "\u2581", "\ud83d\udcaa", "\u2581", "\u2728", "\u2581", "\ud83d\udc49", "\u2581", "\ud83d\udc40", "\u2581", "\ud83d\udcaf", "\u2581", "\ud83c\udf88", "\u2581", "\ud83d\ude48", "\u2581", "\ud83d\ude4c", "\u2581", "\ud83d\udc80", "\u2581", "\ud83d\udc47", "\u2581", "\ud83d\udc4b", "\u2581\u2705", "\u2581", "\ud83c\udf81", "\u2581", "\ud83c\udf1e", "\u2581", "\ud83c\udf38", "\u2581", "\ud83d\udcb0"],
      ids: [1, 97504, 507, 117545, 507, 123057, 507, 96353, 507, 123058, 507, 123169, 507, 121772, 109976, 115475, 507, 122874, 507, 124017, 507, 123983, 507, 123571, 507, 122632, 49509, 25377, 507, 123614, 507, 124105, 507, 124077, 507, 123384, 507, 124382, 507, 123340, 507, 123492, 507, 3, 507, 123306, 507, 110119, 507, 122633, 507, 123659, 507, 123765, 507, 125799, 507, 124322, 507, 122878, 507, 125843, 507, 124011, 507, 125021, 88523, 507, 124698, 507, 125612, 507, 123887, 507, 123979, 2],
      decoded: "[CLS] \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c [UNK] \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0[SEP]",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "\u2728", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\udc41", "\ufe0f", "\u2581", "\ud83d\udc71", "\ud83c\udffb", "\u2581", "\ud83d\udd75", "\u2581", "\u2642", "\ufe0f", "\u2581", "\ud83e\uddd9", "\ud83c\udffb", "\u2581", "\u2642", "\u2581", "\ud83d\udc68", "\ud83c\udffb", "\u2581", "\ud83c\udf3e", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83d\udc69", "\u2581\u2764", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc67", "\u2581", "\ud83d\udc66", "\u2581", "\ud83e\uddd1", "\ud83c\udffb", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1", "\ud83c\udffb", "\u2581", "\ud83c\udff4", "\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f", "\u2581", "\ud83d\udc68", "\ud83c\udffb", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\ud83c\udffc"],
      ids: [1, 507, 110119, 507, 123983, 507, 127294, 25377, 507, 3, 108391, 507, 3, 507, 117868, 25377, 507, 3, 108391, 507, 117868, 507, 125199, 108391, 507, 3, 507, 3, 507, 3, 507, 3, 507, 124709, 49509, 507, 124327, 507, 125199, 507, 124709, 507, 124709, 507, 126640, 507, 126853, 507, 3, 108391, 507, 3, 507, 3, 108391, 507, 126132, 3, 507, 125199, 108391, 49509, 25377, 507, 124327, 507, 125199, 118155, 2],
      decoded: "[CLS] \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f [UNK]\ud83c\udffb [UNK] \u2642\ufe0f [UNK]\ud83c\udffb \u2642 \ud83d\udc68\ud83c\udffb [UNK] [UNK] [UNK] [UNK] \ud83d\udc69 \u2764 \ud83d\udc8b \ud83d\udc68 \ud83d\udc69 \ud83d\udc69 \ud83d\udc67 \ud83d\udc66 [UNK]\ud83c\udffb [UNK] [UNK]\ud83c\udffb \ud83c\udff4[UNK] \ud83d\udc68\ud83c\udffb \u2764\ufe0f \ud83d\udc8b \ud83d\udc68\ud83c\udffc[SEP]",
    },
    CHINESE_LATIN_MIXED: {
      text: BERT_TEST_STRINGS.CHINESE_LATIN_MIXED,
      tokens: ["\u2581a", "h", "\u535a", "\u63a8", "zz"],
      ids: [1, 266, 1537, 122598, 111743, 23260, 2],
      decoded: "[CLS] ah\u535a\u63a8zz[SEP]",
    },
    SIMPLE_WITH_ACCENTS: {
      text: BERT_TEST_STRINGS.SIMPLE_WITH_ACCENTS,
      tokens: ["\u2581H\u00e9", "llo"],
      ids: [1, 93519, 25341, 2],
      decoded: "[CLS] H\u00e9llo[SEP]",
    },
    MIXED_CASE_WITHOUT_ACCENTS: {
      text: BERT_TEST_STRINGS.MIXED_CASE_WITHOUT_ACCENTS,
      tokens: ["\u2581He", "LL", "o", "!", "how", "\u2581Are", "\u2581yo", "U", "?"],
      ids: [1, 383, 17145, 795, 300, 5608, 1396, 14469, 2628, 302, 2],
      decoded: "[CLS] HeLLo!how Are yoU?[SEP]",
    },
    MIXED_CASE_WITH_ACCENTS: {
      text: BERT_TEST_STRINGS.MIXED_CASE_WITH_ACCENTS,
      tokens: ["\u2581H\u00e4", "LL", "o", "!", "how", "\u2581Are", "\u2581yo", "U", "?"],
      ids: [1, 62693, 17145, 795, 300, 5608, 1396, 14469, 2628, 302, 2],
      decoded: "[CLS] H\u00e4LLo!how Are yoU?[SEP]",
    },
  },
  "Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["\u2581How", "\u2581are", "\u2581you", "\u2581do", "ing", "?"],
      ids: [1, 5101, 419, 522, 343, 348, 292, 2],
      decoded: "[CLS] How are you doing?[SEP]",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["\u2581", "0123456789", "\u25810", "\u25811", "\u25812", "\u25813", "\u25814", "\u25815", "\u25816", "\u25817", "\u25818", "\u25819", "\u258110", "\u2581100", "\u25811000"],
      ids: [1, 260, 170160, 498, 334, 357, 382, 420, 431, 571, 618, 631, 775, 476, 967, 3884, 2],
      decoded: "[CLS] 0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000[SEP]",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["\u2581The", "\u2581company", "\u2581was", "\u2581found", "ed", "\u2581in", "\u25812016."],
      ids: [1, 487, 5836, 640, 5898, 346, 282, 13792, 2],
      decoded: "[CLS] The company was founded in 2016.[SEP]",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["\u2581A", "\u2581", "'", "ll", "\u2581", "!!", "to", "?", "'", "d", "''", "d", "\u2581of", ",", "\u2581can", "'", "t", "."],
      ids: [1, 299, 260, 278, 1579, 260, 1524, 477, 292, 278, 286, 4461, 286, 305, 262, 739, 278, 271, 261, 2],
      decoded: "[CLS] A 'll!!to?'d''d of, can't.[SEP]",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["\u2581de", "f", "\u2581main", "():", "\u2581pass"],
      ids: [1, 270, 368, 4398, 78612, 4748, 2],
      decoded: "[CLS] def main(): pass[SEP]",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["\u2581let", "\u2581", "a", "\u2581", "=", "\u2581obj", ".", "toString", "();", "\u2581", "toString", "();"],
      ids: [1, 3257, 260, 263, 260, 350, 50670, 261, 64577, 1994, 260, 64577, 1994, 2],
      decoded: "[CLS] let a = obj.toString(); toString();[SEP]",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["\u2581This", "\u2581is", "\u2581", "a", "\u2581test", "."],
      ids: [1, 1495, 340, 260, 263, 2979, 261, 2],
      decoded: "[CLS] This is a test.[SEP]",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["\u2581UN", "wan", "t\u00e9", "d", ",", "running"],
      ids: [1, 10970, 3016, 3986, 286, 262, 170565, 2],
      decoded: "[CLS] UNwant\u00e9d,running[SEP]",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u2581", "\u751f\u6d3b\u7684", "\u771f", "\u8c1b", "\u662f"],
      ids: [1, 260, 197263, 7275, 241962, 1544, 2],
      decoded: "[CLS] \u751f\u6d3b\u7684\u771f\u8c1b\u662f[SEP]",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u2581", "leading", "\u2581space"],
      ids: [1, 260, 22120, 11496, 2],
      decoded: "[CLS] leading space[SEP]",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["\u2581trail", "ing", "\u2581space"],
      ids: [1, 66699, 348, 11496, 2],
      decoded: "[CLS] trailing space[SEP]",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["\u2581test", "\u2581$1", "\u2581R", "2", "\u2581#3", "\u2581\u20ac4", "\u2581\u00a35", "\u2581\u00a5", "6", "\u2581", "\u20a3", "7", "\u2581\u20b9", "8", "\u2581", "\u20b1", "9", "\u2581test"],
      ids: [1, 2979, 21793, 532, 339, 19403, 157186, 156260, 33481, 452, 260, 242687, 488, 39568, 450, 260, 211232, 496, 2979, 2],
      decoded: "[CLS] test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test[SEP]",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["\u2581I", "\u2581b", "ought", "\u2581an", "\u2581apple", "\u2581for", "\u2581$", "1.00", "\u2581at", "\u2581the", "\u2581store", "."],
      ids: [1, 337, 331, 22280, 462, 44791, 333, 1161, 42645, 345, 288, 5318, 261, 2],
      decoded: "[CLS] I bought an apple for $1.00 at the store.[SEP]",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["\u2581you", "..."],
      ids: [1, 522, 303, 2],
      decoded: "[CLS] you...[SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["\u2581you", "..."],
      ids: [1, 522, 303, 2],
      decoded: "[CLS] you...[SEP]",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["\u2581you", "...", "\u2581you", "..."],
      ids: [1, 522, 303, 522, 303, 2],
      decoded: "[CLS] you... you...[SEP]",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["\u2581w", "eird", "\u2581", "\uff5e", "\u2581edge", "\u2581", "\uff5e", "\u2581case"],
      ids: [1, 415, 116640, 260, 2790, 53876, 260, 2790, 4073, 2],
      decoded: "[CLS] weird \uff5e edge \uff5e case[SEP]",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u2581This", "\u2581is", "\u2581", "a", "\u2581test", "\u2581", "."],
      ids: [1, 1495, 340, 260, 263, 2979, 260, 261, 2],
      decoded: "[CLS] This is a test.[SEP]",
    },
    POPULAR_EMOJIS: {
      text: BASE_TEST_STRINGS.POPULAR_EMOJIS,
      tokens: ["\u2581", "\ud83d\ude02", "\u2581", "\ud83d\udc4d", "\u2581", "\ud83e\udd23", "\u2581", "\ud83d\ude0d", "\u2581", "\ud83d\ude2d", "\u2581", "\ud83c\udf89", "\u2581", "\ud83d\ude4f", "\u2581", "\ud83d\ude0a", "\u2581", "\ud83d\udd25", "\u2581", "\ud83d\ude01", "\u2581", "\ud83d\ude05", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\ude06", "\u2581", "\ud83d\udc4f", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc9c", "\u2581", "\ud83d\udc9a", "\u2581", "\ud83d\udc97", "\u2581", "\ud83d\udc99", "\u2581", "\ud83d\udda4", "\u2581", "\ud83d\ude0e", "\u2581", "\ud83d\udc4c", "\u2581", "\ud83e\udd73", "\u2581", "\ud83d\udcaa", "\u2581", "\u2728", "\u2581\ud83d\udc49", "\u2581", "\ud83d\udc40", "\u2581", "\ud83d\udcaf", "\u2581", "\ud83c\udf88", "\u2581", "\ud83d\ude48", "\u2581", "\ud83d\ude4c", "\u2581", "\ud83d\udc80", "\u2581", "\ud83d\udc47", "\u2581", "\ud83d\udc4b", "\u2581\u2705", "\u2581", "\ud83c\udf81", "\u2581", "\ud83c\udf1e", "\u2581", "\ud83c\udf38", "\u2581", "\ud83d\udcb0"],
      ids: [1, 260, 116844, 260, 72330, 260, 160951, 260, 78796, 260, 180546, 260, 212774, 260, 102930, 260, 71509, 260, 96089, 260, 137652, 260, 194608, 260, 182033, 260, 164467, 260, 149267, 56787, 4668, 260, 210251, 260, 195202, 260, 178523, 260, 167604, 260, 236081, 260, 157800, 260, 162843, 260, 242580, 260, 174590, 260, 65271, 113700, 260, 239652, 260, 237474, 260, 240937, 260, 239131, 260, 216701, 260, 242618, 260, 133395, 260, 240645, 82147, 260, 49599, 260, 239888, 260, 152102, 260, 239168, 2],
      decoded: "[CLS] \ud83d\ude02 \ud83d\udc4d \ud83e\udd23 \ud83d\ude0d \ud83d\ude2d \ud83c\udf89 \ud83d\ude4f \ud83d\ude0a \ud83d\udd25 \ud83d\ude01 \ud83d\ude05 \ud83e\udd17 \ud83d\ude06 \ud83d\udc4f \u2764\ufe0f \ud83d\udc9c \ud83d\udc9a \ud83d\udc97 \ud83d\udc99 \ud83d\udda4 \ud83d\ude0e \ud83d\udc4c \ud83e\udd73 \ud83d\udcaa \u2728 \ud83d\udc49 \ud83d\udc40 \ud83d\udcaf \ud83c\udf88 \ud83d\ude48 \ud83d\ude4c \ud83d\udc80 \ud83d\udc47 \ud83d\udc4b \u2705 \ud83c\udf81 \ud83c\udf1e \ud83c\udf38 \ud83d\udcb0[SEP]",
    },
    MULTIBYTE_EMOJIS: {
      text: BASE_TEST_STRINGS.MULTIBYTE_EMOJIS,
      tokens: ["\u2581", "\u2728", "\u2581", "\ud83e\udd17", "\u2581", "\ud83d\udc41", "\ufe0f", "\u2581", "\ud83d\udc71", "\ud83c\udffb", "\u2581", "\ud83d\udd75", "\u2581", "\u2642", "\ufe0f", "\u2581", "\ud83e\uddd9", "\ud83c\udffb", "\u2581", "\u2642", "\u2581", "\ud83d\udc68", "\ud83c\udffb", "\u2581", "\ud83c\udf3e", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1", "\u2581", "\ud83d\udc69", "\u2581\u2764", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc69", "\u2581", "\ud83d\udc67", "\u2581", "\ud83d\udc66", "\u2581", "\ud83e\uddd1", "\ud83c\udffb", "\u2581", "\ud83e\udd1d", "\u2581", "\ud83e\uddd1", "\ud83c\udffb", "\u2581", "\ud83c\udff4", "\udb40\udc67", "\udb40\udc62", "\udb40\udc65", "\udb40\udc6e", "\udb40\udc67", "\udb40\udc7f", "\u2581", "\ud83d\udc68", "\ud83c\udffb", "\u2581\u2764", "\ufe0f", "\u2581", "\ud83d\udc8b", "\u2581", "\ud83d\udc68", "\ud83c\udffc"],
      ids: [1, 260, 65271, 260, 182033, 260, 16307, 4668, 260, 244774, 75846, 260, 247133, 260, 50622, 4668, 260, 3, 75846, 260, 50622, 260, 239432, 75846, 260, 243052, 260, 244250, 260, 243394, 260, 244250, 260, 239098, 56787, 260, 223802, 260, 239432, 260, 239098, 260, 239098, 260, 241727, 260, 242446, 260, 244250, 75846, 260, 243394, 260, 244250, 75846, 260, 244177, 245994, 247023, 248837, 248531, 245994, 245953, 260, 239432, 75846, 56787, 4668, 260, 223802, 260, 239432, 159667, 2],
      decoded: "[CLS] \u2728 \ud83e\udd17 \ud83d\udc41\ufe0f \ud83d\udc71\ud83c\udffb \ud83d\udd75 \u2642\ufe0f [UNK]\ud83c\udffb \u2642 \ud83d\udc68\ud83c\udffb \ud83c\udf3e \ud83e\uddd1 \ud83e\udd1d \ud83e\uddd1 \ud83d\udc69 \u2764 \ud83d\udc8b \ud83d\udc68 \ud83d\udc69 \ud83d\udc69 \ud83d\udc67 \ud83d\udc66 \ud83e\uddd1\ud83c\udffb \ud83e\udd1d \ud83e\uddd1\ud83c\udffb \ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f \ud83d\udc68\ud83c\udffb \u2764\ufe0f \ud83d\udc8b \ud83d\udc68\ud83c\udffc[SEP]",
    },
  },
};