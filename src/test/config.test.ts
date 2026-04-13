import test from "node:test";
import assert from "node:assert/strict";

import { loadResolvedConfig } from "../config.js";

test("loadResolvedConfig prefers explicit api key flag", async () => {
  const previous = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "env-value";

  try {
    const config = await loadResolvedConfig("/tmp/transcribe-cli-test-config.toml", "flag-value");
    assert.equal(config.providers.openai.apiKey, "flag-value");
    assert.equal(config.providers.openai.authSource, "flag");
  } finally {
    if (previous) {
      process.env.OPENAI_API_KEY = previous;
    } else {
      delete process.env.OPENAI_API_KEY;
    }
  }
});
