import test from "node:test";
import assert from "node:assert/strict";

import {
  defaultResponseFormatForTranscribe,
  mergeBatchResults,
  validateTranscriptionOptions,
} from "../audio.js";
import type { BatchChunkResult } from "../types.js";

test("defaultResponseFormatForTranscribe prefers diarized_json for diarization model", () => {
  const format = defaultResponseFormatForTranscribe({
    json: false,
    model: "gpt-4o-transcribe-diarize",
    timestampGranularities: [],
  });

  assert.equal(format, "diarized_json");
});

test("validateTranscriptionOptions rejects timestamp granularities on non-whisper model", () => {
  assert.throws(() => {
    validateTranscriptionOptions({
      model: "gpt-4o-mini-transcribe",
      responseFormat: "json",
      timestampGranularities: ["word"],
      knownSpeakers: [],
    });
  });
});

test("mergeBatchResults shifts diarized segment timing", () => {
  const chunks: BatchChunkResult[] = [
    {
      index: 0,
      path: "/tmp/chunk-000.mp3",
      offsetSeconds: 0,
      durationSeconds: 3,
      outputPath: "/tmp/out-000.json",
      result: {
        command: "transcribe",
        provider: "openai",
        model: "gpt-4o-transcribe-diarize",
        audioPath: "/tmp/chunk-000.mp3",
        responseFormat: "diarized_json",
        transcriptText: "Hello there",
        data: {
          segments: [{ speaker: "A", start: 0, end: 1.5, text: "Hello there" }],
        },
      },
    },
    {
      index: 1,
      path: "/tmp/chunk-001.mp3",
      offsetSeconds: 3,
      durationSeconds: 4,
      outputPath: "/tmp/out-001.json",
      result: {
        command: "transcribe",
        provider: "openai",
        model: "gpt-4o-transcribe-diarize",
        audioPath: "/tmp/chunk-001.mp3",
        responseFormat: "diarized_json",
        transcriptText: "General Kenobi",
        data: {
          segments: [{ speaker: "B", start: 0.25, end: 1.75, text: "General Kenobi" }],
        },
      },
    },
  ];

  const merged = mergeBatchResults(chunks, "diarized_json");
  const segments = merged.segments as Array<{ start: number; end: number }>;

  assert.equal(segments.length, 2);
  assert.equal(segments[1]?.start, 3.25);
  assert.equal(segments[1]?.end, 4.75);
});
