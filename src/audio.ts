import { createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import { homedir } from "node:os";
import { spawn } from "node:child_process";

import { AppError } from "./errors.js";
import type {
  AudioCommandResult,
  BatchChunkResult,
  OutputResponseFormat,
  TimestampGranularity,
} from "./types.js";

export const MAX_AUDIO_BYTES = 25 * 1024 * 1024;
export const INPUT_FORMATS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"];
export const ALL_RESPONSE_FORMATS = [
  "json",
  "text",
  "srt",
  "verbose_json",
  "vtt",
  "diarized_json",
] as const;

export interface AudioInspection {
  path: string;
  sizeBytes: number;
  durationSeconds?: number;
  formatName?: string;
  bitRate?: number;
}

function runCommand(command: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => {
      if (code === 0) {
        resolvePromise({ stdout, stderr });
        return;
      }
      reject(new AppError("command_failed", `${command} exited with code ${code}`, 1, { stderr }));
    });
  });
}

export function commandOnPath(command: string): boolean {
  const path = process.env.PATH || "";
  return path.split(":").some((part) => existsSync(join(part, command)));
}

export function createFileStream(path: string) {
  return createReadStream(path);
}

export async function ensureAudioFile(path: string): Promise<void> {
  const resolved = resolve(path);
  if (!existsSync(resolved)) {
    throw new AppError("missing_audio", `Audio file not found: ${path}`);
  }
  const fileStat = await stat(resolved);
  if (!fileStat.isFile()) {
    throw new AppError("invalid_audio", `Not a file: ${path}`);
  }
}

export async function ensureAudioFiles(paths: string[]): Promise<void> {
  for (const path of paths) {
    await ensureAudioFile(path);
  }
}

export async function inspectAudio(path: string): Promise<AudioInspection> {
  await ensureAudioFile(path);
  const fileStat = await stat(path);
  if (!commandOnPath("ffprobe")) {
    return {
      path,
      sizeBytes: fileStat.size,
    };
  }

  try {
    const { stdout } = await runCommand("ffprobe", [
      "-v",
      "error",
      "-show_entries",
      "format=duration,size,format_name,bit_rate",
      "-of",
      "json",
      path,
    ]);
    const payload = JSON.parse(stdout) as {
      format?: { duration?: string; size?: string; format_name?: string; bit_rate?: string };
    };
    return {
      path,
      sizeBytes: Number.parseInt(payload.format?.size || `${fileStat.size}`, 10),
      durationSeconds: payload.format?.duration ? Number(payload.format.duration) : undefined,
      formatName: payload.format?.format_name,
      bitRate: payload.format?.bit_rate ? Number(payload.format.bit_rate) : undefined,
    };
  } catch {
    return {
      path,
      sizeBytes: fileStat.size,
    };
  }
}

export function inferOutputExtension(format: OutputResponseFormat): string {
  switch (format) {
    case "text":
      return "txt";
    case "json":
    case "verbose_json":
    case "diarized_json":
      return "json";
    case "srt":
      return "srt";
    case "vtt":
      return "vtt";
  }
}

export function defaultResponseFormatForTranscribe(args: {
  json: boolean;
  model: string;
  responseFormat?: OutputResponseFormat;
  timestampGranularities: TimestampGranularity[];
}): OutputResponseFormat {
  if (args.responseFormat) {
    return args.responseFormat;
  }
  if (args.model.includes("transcribe-diarize")) {
    return "diarized_json";
  }
  if (args.timestampGranularities.length > 0) {
    return "verbose_json";
  }
  return args.json ? "json" : "text";
}

export function defaultResponseFormatForTranslate(args: {
  json: boolean;
  responseFormat?: OutputResponseFormat;
}): OutputResponseFormat {
  return args.responseFormat || (args.json ? "json" : "text");
}

export function validateResponseFormatForModel(model: string, format: OutputResponseFormat): void {
  if (model === "whisper-1") {
    if (!["json", "text", "srt", "verbose_json", "vtt"].includes(format)) {
      throw new AppError("invalid_response_format", `${format} is not supported for whisper-1.`);
    }
    return;
  }
  if (model === "gpt-4o-transcribe-diarize") {
    if (!["json", "text", "diarized_json"].includes(format)) {
      throw new AppError("invalid_response_format", `${format} is not supported for ${model}.`);
    }
    return;
  }
  if (!["json", "text"].includes(format)) {
    throw new AppError("invalid_response_format", `${format} is not supported for ${model}.`);
  }
}

export function validateTranscriptionOptions(args: {
  model: string;
  responseFormat: OutputResponseFormat;
  prompt?: string;
  includeLogprobs?: boolean;
  timestampGranularities: TimestampGranularity[];
  knownSpeakers: string[];
}): void {
  validateResponseFormatForModel(args.model, args.responseFormat);

  if (args.timestampGranularities.length > 0 && args.model !== "whisper-1") {
    throw new AppError(
      "invalid_timestamp_model",
      "Timestamp granularities are only supported for whisper-1.",
    );
  }
  if (args.timestampGranularities.length > 0 && args.responseFormat !== "verbose_json") {
    throw new AppError(
      "invalid_timestamp_format",
      "Timestamp granularities require response format verbose_json.",
    );
  }
  if (
    args.includeLogprobs &&
    !["gpt-4o-transcribe", "gpt-4o-mini-transcribe"].includes(args.model)
  ) {
    throw new AppError(
      "invalid_logprobs_model",
      "Logprobs are only supported for gpt-4o-transcribe and gpt-4o-mini-transcribe.",
    );
  }
  if (args.prompt && args.model === "gpt-4o-transcribe-diarize") {
    throw new AppError(
      "invalid_prompt_model",
      "Prompts are not supported for gpt-4o-transcribe-diarize.",
    );
  }
  if (args.knownSpeakers.length > 0 && args.model !== "gpt-4o-transcribe-diarize") {
    throw new AppError(
      "invalid_known_speakers_model",
      "Known speaker references are only supported for gpt-4o-transcribe-diarize.",
    );
  }
  if (args.responseFormat === "diarized_json" && args.model !== "gpt-4o-transcribe-diarize") {
    throw new AppError(
      "invalid_diarized_model",
      "diarized_json requires gpt-4o-transcribe-diarize.",
    );
  }
}

export function parseNumberOption(label: string, value: string): number {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    throw new AppError("invalid_option", `Invalid number for ${label}: ${value}`);
  }
  return parsed;
}

export function collect(value: string, previous: string[] = []): string[] {
  previous.push(value);
  return previous;
}

export function parseKeyValuePair(value: string): { key: string; value: string } {
  const index = value.indexOf("=");
  if (index === -1) {
    throw new AppError("invalid_option", `Expected key=value, got: ${value}`);
  }
  const key = value.slice(0, index).trim();
  const parsedValue = value.slice(index + 1).trim();
  if (!key) {
    throw new AppError("invalid_option", `Missing key in query pair: ${value}`);
  }
  return { key, value: parsedValue };
}

export function buildOutputPath(args: {
  audioPath: string;
  responseFormat: OutputResponseFormat;
  out?: string;
  outDir?: string;
}): string {
  const extension = inferOutputExtension(args.responseFormat);
  if (args.out) {
    if (extname(args.out)) {
      return resolve(args.out);
    }
    return resolve(`${args.out}.${extension}`);
  }
  if (args.outDir) {
    return resolve(
      args.outDir,
      `${basename(args.audioPath, extname(args.audioPath))}.transcript.${extension}`,
    );
  }
  return resolve(`${basename(args.audioPath, extname(args.audioPath))}.transcript.${extension}`);
}

export async function writeTranscriptOutput(args: {
  audioPath: string;
  responseFormat: OutputResponseFormat;
  out?: string;
  outDir?: string;
  content: string;
}): Promise<string> {
  const outputPath = buildOutputPath(args);
  await mkdir(dirnameSafe(outputPath), { recursive: true });
  await writeFile(outputPath, args.content, "utf8");
  return outputPath;
}

function dirnameSafe(path: string): string {
  const lastSlash = path.lastIndexOf("/");
  return lastSlash === -1 ? "." : path.slice(0, lastSlash) || "/";
}

export function stringifyTranscriptData(result: AudioCommandResult): string {
  if (
    result.responseFormat === "text" ||
    result.responseFormat === "srt" ||
    result.responseFormat === "vtt"
  ) {
    return result.transcriptText;
  }
  return JSON.stringify(result.data, null, 2);
}

function shiftTimeFields(items: Record<string, unknown>[], offsetSeconds: number) {
  return items.map((item) => {
    const copy: Record<string, unknown> = { ...item };
    if (typeof copy.start === "number") {
      copy.start = copy.start + offsetSeconds;
    }
    if (typeof copy.end === "number") {
      copy.end = copy.end + offsetSeconds;
    }
    return copy;
  });
}

export function mergeBatchResults(
  chunks: BatchChunkResult[],
  responseFormat: OutputResponseFormat,
): Record<string, unknown> {
  const mergedText = chunks
    .map((chunk) => chunk.result.transcriptText.trim())
    .join("\n\n")
    .trim();

  if (responseFormat === "text" || responseFormat === "srt" || responseFormat === "vtt") {
    return {
      response_format: responseFormat,
      text: mergedText,
      chunks: chunks.map((chunk) => ({
        index: chunk.index,
        path: chunk.path,
        offset_seconds: chunk.offsetSeconds,
        duration_seconds: chunk.durationSeconds,
        output_path: chunk.outputPath,
      })),
    };
  }

  const merged = {
    response_format: responseFormat,
    text: mergedText,
    chunks: chunks.map((chunk) => ({
      index: chunk.index,
      path: chunk.path,
      offset_seconds: chunk.offsetSeconds,
      duration_seconds: chunk.durationSeconds,
      output_path: chunk.outputPath,
      result: chunk.result.data,
    })),
    segments: [] as Record<string, unknown>[],
    words: [] as Record<string, unknown>[],
    logprobs: [] as unknown[],
  };

  for (const chunk of chunks) {
    const data = chunk.result.data as Record<string, unknown>;
    if (Array.isArray(data.segments)) {
      merged.segments.push(
        ...shiftTimeFields(data.segments as Record<string, unknown>[], chunk.offsetSeconds),
      );
    }
    if (Array.isArray(data.words)) {
      merged.words.push(
        ...shiftTimeFields(data.words as Record<string, unknown>[], chunk.offsetSeconds),
      );
    }
    if (Array.isArray(data.logprobs)) {
      merged.logprobs.push(...data.logprobs);
    }
  }

  return merged;
}

export async function createSpeechWorkingCopy(args: {
  sourcePath: string;
  outputPath: string;
  sampleRate: number;
  bitrate: string;
}): Promise<void> {
  if (!commandOnPath("ffmpeg")) {
    throw new AppError(
      "missing_ffmpeg",
      "ffmpeg is required to create a speech-friendly working copy.",
    );
  }
  await mkdir(dirnameSafe(args.outputPath), { recursive: true });
  await runCommand("ffmpeg", [
    "-hide_banner",
    "-y",
    "-i",
    args.sourcePath,
    "-ac",
    "1",
    "-ar",
    String(args.sampleRate),
    "-c:a",
    "libmp3lame",
    "-b:a",
    args.bitrate,
    args.outputPath,
  ]);
}

export async function segmentAudio(args: {
  sourcePath: string;
  outDir: string;
  segmentSeconds: number;
}): Promise<string[]> {
  if (!commandOnPath("ffmpeg")) {
    throw new AppError("missing_ffmpeg", "ffmpeg is required to split audio into chunks.");
  }
  await mkdir(args.outDir, { recursive: true });
  await runCommand("ffmpeg", [
    "-hide_banner",
    "-y",
    "-i",
    args.sourcePath,
    "-f",
    "segment",
    "-segment_time",
    String(args.segmentSeconds),
    "-c",
    "copy",
    join(args.outDir, "chunk-%03d.mp3"),
  ]);
  const entries = await readdir(args.outDir);
  return entries
    .filter((entry) => entry.startsWith("chunk-"))
    .sort()
    .map((entry) => join(args.outDir, entry));
}

export function expandHome(path: string): string {
  if (path === "~") {
    return homedir();
  }
  if (path.startsWith("~/")) {
    return join(homedir(), path.slice(2));
  }
  return path;
}

export async function fileToDataUrl(path: string): Promise<string> {
  const bytes = await readFile(path);
  const ext = extname(path).toLowerCase();
  const mimeType =
    ext === ".mp3"
      ? "audio/mpeg"
      : ext === ".m4a"
        ? "audio/mp4"
        : ext === ".wav"
          ? "audio/wav"
          : ext === ".webm"
            ? "audio/webm"
            : "application/octet-stream";
  return `data:${mimeType};base64,${bytes.toString("base64")}`;
}
