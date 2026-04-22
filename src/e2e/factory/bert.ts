// SPDX-License-Identifier: MIT

import { AutoTokenizer } from "@huggingface/transformers";

import { LLMLingua2 } from "../../index.js";
import { EXAMPLES } from "../long-texts.js";

const modelName = "Arcoldd/llmlingua4j-bert-base-onnx";
const oai_tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4o");

const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(modelName, {
  transformersJSConfig: {
    device: "auto",
    dtype: "fp32",
  },
  oaiTokenizer: oai_tokenizer,
  modelSpecificOptions: {
    subfolder: "",
  },
});

const start = performance.now();

const result = await promptCompressor.compress_prompt(
  EXAMPLES[EXAMPLES.length - 1],
  {
    rate: 0.5,
  },
);

const end = performance.now();

console.log({ result });

console.log("Time taken for compression:", end - start, "ms");
console.log(
  "Time taken for compression (human-readable):",
  ((end - start) / 1000).toFixed(2),
  "seconds",
);
