import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCarryoverPrompt,
  defaultResponseFormatForTranscribe,
  mergeBatchResults,
  planChunkRanges,
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

test("planChunkRanges prefers nearby silence boundaries", () => {
  const ranges = planChunkRanges({
    durationSeconds: 25,
    targetSeconds: 10,
    maxChunkSeconds: 14,
    boundaryWindowSeconds: 3,
    boundaries: [{ end: 11.8 }, { end: 20.9 }],
    forceChunk: true,
  });

  assert.equal(ranges.length, 3);
  assert.equal(ranges[0]?.endSeconds, 11.8);
  assert.equal(ranges[1]?.startSeconds, 11.8);
});

test("buildCarryoverPrompt adds previous transcript context", () => {
  const prompt = buildCarryoverPrompt({
    model: "gpt-4o-mini-transcribe",
    previousTranscript: "Earlier chunk text here.",
  });

  assert.match(prompt || "", /Earlier chunk text here/);
  assert.match(prompt || "", /Continue this transcript smoothly/);
});
