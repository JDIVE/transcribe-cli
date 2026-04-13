#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import { Command, Option } from "commander";

import {
  ALL_RESPONSE_FORMATS,
  buildCarryoverPrompt,
  collect,
  commandOnPath,
  createSpeechWorkingCopy,
  DEFAULT_BATCH_CHUNK_SECONDS,
  DEFAULT_BOUNDARY_WINDOW_SECONDS,
  DEFAULT_MIN_SILENCE_SECONDS,
  DEFAULT_SILENCE_THRESHOLD_DB,
  defaultResponseFormatForTranscribe,
  defaultResponseFormatForTranslate,
  detectSilenceBoundaries,
  ensureAudioFiles,
  estimateSafeChunkSeconds,
  expandHome,
  extractAudioChunks,
  inspectAudio,
  MAX_AUDIO_BYTES,
  mergeBatchResults,
  modelSupportsPrompt,
  parseKeyValuePair,
  parseNumberOption,
  planChunkRanges,
  stringifyTranscriptData,
  validateResponseFormatForModel,
  validateTranscriptionOptions,
  writeTranscriptOutput,
} from "./audio.js";
import { ensureConfigDir, loadResolvedConfig, writeStarterConfig } from "./config.js";
import { AppError, toErrorPayload } from "./errors.js";
import { printHuman, printJson, successEnvelope } from "./output.js";
import type { ProviderAdapter } from "./providers/index.js";
import { OpenAIProvider } from "./providers/openai.js";
import type {
  AudioCommandOptions,
  BatchChunkResult,
  OutputResponseFormat,
  ProviderName,
  TimestampGranularity,
} from "./types.js";

const VERSION = "0.1.3";

const providers: Record<ProviderName, ProviderAdapter> = {
  openai: new OpenAIProvider(),
};

function getJsonFlag(): boolean {
  return process.argv.includes("--json");
}

async function withConfig<T>(
  action: (args: {
    json: boolean;
    config: Awaited<ReturnType<typeof loadResolvedConfig>>;
  }) => Promise<T>,
  command: Command,
): Promise<T> {
  const opts = command.optsWithGlobals() as { config?: string; json?: boolean; apiKey?: string };
  const config = await loadResolvedConfig(opts.config, opts.apiKey);
  return action({ json: Boolean(opts.json), config });
}

function requireProvider(name: ProviderName): ProviderAdapter {
  return providers[name];
}

function selectProvider(
  command: Command,
  config: Awaited<ReturnType<typeof loadResolvedConfig>>,
): ProviderName {
  const opts = command.optsWithGlobals() as { provider?: ProviderName };
  return opts.provider || config.defaultProvider;
}

function collectQueryPairs(values: string[]): Record<string, string | string[]> {
  const result: Record<string, string | string[]> = {};
  for (const value of values) {
    const pair = parseKeyValuePair(value);
    const current = result[pair.key];
    if (!current) {
      result[pair.key] = pair.value;
      continue;
    }
    if (Array.isArray(current)) {
      current.push(pair.value);
      result[pair.key] = current;
      continue;
    }
    result[pair.key] = [current, pair.value];
  }
  return result;
}

function renderDoctorHuman(data: {
  configPath: string;
  configExists: boolean;
  defaultProvider: ProviderName;
  providers: Array<{
    provider: ProviderName;
    authSource: string;
    configured: boolean;
    reachable?: boolean;
  }>;
  tools: Record<string, boolean>;
}): void {
  printHuman("Doctor", [
    `config: ${data.configPath} (${data.configExists ? "present" : "missing"})`,
    `default provider: ${data.defaultProvider}`,
    ...data.providers.map(
      (provider) =>
        `${provider.provider}: auth=${provider.authSource} configured=${provider.configured} reachable=${provider.reachable ?? "n/a"}`,
    ),
    `ffmpeg: ${data.tools.ffmpeg ? "found" : "missing"}`,
    `ffprobe: ${data.tools.ffprobe ? "found" : "missing"}`,
  ]);
}

function pickTranscriptionModel(
  command: Command,
  config: Awaited<ReturnType<typeof loadResolvedConfig>>,
): string {
  const opts = command.optsWithGlobals() as {
    model?: string;
    diarize?: boolean;
    provider?: ProviderName;
  };
  if (opts.model) {
    return opts.model;
  }
  if (opts.diarize) {
    return config.providers.openai.defaultDiarizationModel;
  }
  return config.providers.openai.defaultTranscriptionModel;
}

function buildAudioOptions(args: {
  command: Command;
  config: Awaited<ReturnType<typeof loadResolvedConfig>>;
  audioPath: string;
  mode: "transcribe" | "translate";
  json: boolean;
}): AudioCommandOptions {
  const opts = args.command.optsWithGlobals() as {
    provider?: ProviderName;
    model?: string;
    responseFormat?: OutputResponseFormat;
    language?: string;
    prompt?: string;
    temperature?: number;
    includeLogprobs?: boolean;
    timestamp?: TimestampGranularity[];
    chunkingStrategy?: string;
    knownSpeaker?: string[];
    dryRun?: boolean;
    diarize?: boolean;
  };

  if (args.mode === "translate") {
    return {
      provider: opts.provider || args.config.defaultProvider,
      audioPath: args.audioPath,
      model: opts.model || args.config.providers.openai.defaultTranslationModel,
      responseFormat: defaultResponseFormatForTranslate({
        json: args.json,
        responseFormat: opts.responseFormat,
      }),
      language: opts.language,
      prompt: opts.prompt,
      temperature: opts.temperature,
      dryRun: opts.dryRun,
    };
  }

  const model = pickTranscriptionModel(args.command, args.config);
  const timestampGranularities = opts.timestamp || [];
  const responseFormat = defaultResponseFormatForTranscribe({
    json: args.json,
    model,
    responseFormat: opts.responseFormat,
    timestampGranularities,
  });

  return {
    provider: opts.provider || args.config.defaultProvider,
    audioPath: args.audioPath,
    model,
    responseFormat,
    language: opts.language,
    prompt: opts.prompt,
    temperature: opts.temperature,
    includeLogprobs: opts.includeLogprobs,
    timestampGranularities,
    chunkingStrategy: opts.chunkingStrategy,
    knownSpeakers: opts.knownSpeaker || [],
    dryRun: opts.dryRun,
  };
}

async function runAudioCommand(args: {
  command: Command;
  audioPaths: string[];
  mode: "transcribe" | "translate";
}): Promise<void> {
  await withConfig(async ({ json, config }) => {
    const opts = args.command.optsWithGlobals() as {
      provider?: ProviderName;
      out?: string;
      outDir?: string;
      stdout?: boolean;
    };

    if (opts.out && args.audioPaths.length > 1) {
      throw new AppError("invalid_out", "--out only supports a single audio file.");
    }
    if (opts.stdout && args.audioPaths.length > 1) {
      throw new AppError("invalid_stdout", "--stdout only supports a single audio file.");
    }
    if (opts.stdout && (opts.out || opts.outDir)) {
      throw new AppError("invalid_stdout", "--stdout cannot be combined with --out or --out-dir.");
    }

    await ensureAudioFiles(args.audioPaths);

    const providerName = selectProvider(args.command, config);
    const provider = requireProvider(providerName);
    const results = [];

    for (const audioPath of args.audioPaths) {
      const audioOptions = buildAudioOptions({
        command: args.command,
        config,
        audioPath,
        mode: args.mode,
        json,
      });

      if (args.mode === "transcribe") {
        validateTranscriptionOptions({
          model: audioOptions.model!,
          responseFormat: audioOptions.responseFormat!,
          prompt: audioOptions.prompt,
          includeLogprobs: audioOptions.includeLogprobs,
          timestampGranularities: audioOptions.timestampGranularities || [],
          knownSpeakers: audioOptions.knownSpeakers || [],
        });
      } else {
        validateResponseFormatForModel(audioOptions.model!, audioOptions.responseFormat!);
      }

      const result =
        args.mode === "transcribe"
          ? await provider.transcribe({ config }, audioOptions)
          : await provider.translate({ config }, audioOptions);

      results.push(result);
    }

    if (opts.stdout) {
      process.stdout.write(`${stringifyTranscriptData(results[0]!)}\n`);
      return;
    }

    const outputs = [];
    for (const result of results) {
      if (result.dryRun) {
        outputs.push({
          audioPath: result.audioPath,
          outputPath: null,
          dryRun: true,
          request: result.data,
        });
        continue;
      }
      const outputPath = await writeTranscriptOutput({
        audioPath: result.audioPath,
        responseFormat: result.responseFormat,
        out: opts.out,
        outDir: opts.outDir,
        content: stringifyTranscriptData(result),
      });
      outputs.push({
        audioPath: result.audioPath,
        outputPath,
        responseFormat: result.responseFormat,
        model: result.model,
      });
    }

    const payload = successEnvelope(args.mode, {
      provider: providerName,
      count: results.length,
      outputs,
    });

    if (json) {
      printJson(payload);
      return;
    }

    printHuman(
      args.mode === "transcribe" ? "Transcribed" : "Translated",
      outputs.map((item) =>
        item.dryRun ? `${item.audioPath}: dry-run` : `${item.audioPath} -> ${item.outputPath}`,
      ),
    );
  }, args.command);
}

function defaultBatchWorkingDir(audioPath: string): string {
  const stem = basename(audioPath, extname(audioPath));
  return resolve("transcribe-cli-batch", stem);
}

async function runBatchCommand(command: Command, audioPaths: string[]): Promise<void> {
  await withConfig(async ({ json, config }) => {
    const opts = command.optsWithGlobals() as {
      provider?: ProviderName;
      workingDir?: string;
      chunkSeconds?: number;
      alwaysChunk?: boolean;
      boundaryWindowSeconds?: number;
      minSilenceSeconds?: number;
      silenceThresholdDb?: number;
      promptCarryover?: boolean;
      bitrate?: string;
      sampleRate?: number;
      dryRun?: boolean;
    };

    const providerName = selectProvider(command, config);
    const provider = requireProvider(providerName);
    const batchOutputs: Array<Record<string, unknown>> = [];

    await ensureAudioFiles(audioPaths);

    for (const audioPath of audioPaths) {
      const audioOptions = buildAudioOptions({
        command,
        config,
        audioPath,
        mode: "transcribe",
        json,
      });

      validateTranscriptionOptions({
        model: audioOptions.model!,
        responseFormat: audioOptions.responseFormat!,
        prompt: audioOptions.prompt,
        includeLogprobs: audioOptions.includeLogprobs,
        timestampGranularities: audioOptions.timestampGranularities || [],
        knownSpeakers: audioOptions.knownSpeakers || [],
      });

      const sourceInspection = await inspectAudio(audioPath);
      const jobDir = resolve(
        opts.workingDir ? expandHome(opts.workingDir) : defaultBatchWorkingDir(audioPath),
      );
      const outputsDir = join(jobDir, "chunks");
      const chunkOutputsDir = join(jobDir, "outputs");
      const requestedChunkSeconds = opts.chunkSeconds || DEFAULT_BATCH_CHUNK_SECONDS;
      const sampleRate = opts.sampleRate || 22050;
      const bitrate = opts.bitrate || "64k";
      const boundaryWindowSeconds = opts.boundaryWindowSeconds || DEFAULT_BOUNDARY_WINDOW_SECONDS;
      const minSilenceSeconds = opts.minSilenceSeconds || DEFAULT_MIN_SILENCE_SECONDS;
      const silenceThresholdDb = opts.silenceThresholdDb || DEFAULT_SILENCE_THRESHOLD_DB;
      const promptCarryover =
        opts.promptCarryover !== false && modelSupportsPrompt(audioOptions.model!);
      const sourceNeedsSpeechCopy = sourceInspection.sizeBytes > MAX_AUDIO_BYTES;

      const plan = {
        sourcePath: audioPath,
        jobDir,
        sourceSizeBytes: sourceInspection.sizeBytes,
        sourceDurationSeconds: sourceInspection.durationSeconds,
        sourceOverLimit: sourceInspection.sizeBytes > MAX_AUDIO_BYTES,
        wouldCreateSpeechCopy: sourceNeedsSpeechCopy,
        wouldChunk: sourceInspection.sizeBytes > MAX_AUDIO_BYTES || Boolean(opts.alwaysChunk),
        requestedChunkSeconds,
        boundaryWindowSeconds,
        minSilenceSeconds,
        silenceThresholdDb,
        promptCarryover,
        responseFormat: audioOptions.responseFormat,
        model: audioOptions.model,
      };

      if (opts.dryRun) {
        batchOutputs.push({
          audioPath,
          dryRun: true,
          plan,
        });
        continue;
      }

      await mkdir(jobDir, { recursive: true });
      let workingPath = audioPath;
      let workingInspection = sourceInspection;

      if (sourceNeedsSpeechCopy) {
        workingPath = join(jobDir, "working-copy.mp3");
        await createSpeechWorkingCopy({
          sourcePath: audioPath,
          outputPath: workingPath,
          sampleRate,
          bitrate,
        });
        workingInspection = await inspectAudio(workingPath);
      }

      let chunkPaths = [workingPath];
      let chunkRanges = [
        {
          index: 0,
          startSeconds: 0,
          endSeconds: workingInspection.durationSeconds || 0,
        },
      ];
      if (workingInspection.sizeBytes > MAX_AUDIO_BYTES || opts.alwaysChunk) {
        const safeChunkSeconds = estimateSafeChunkSeconds({
          durationSeconds: workingInspection.durationSeconds,
          sizeBytes: workingInspection.sizeBytes,
          preferredSeconds: requestedChunkSeconds,
        });
        const silenceBoundaries = await detectSilenceBoundaries({
          sourcePath: workingPath,
          thresholdDb: silenceThresholdDb,
          minSilenceSeconds,
        });
        chunkRanges = planChunkRanges({
          durationSeconds: workingInspection.durationSeconds || 0,
          targetSeconds: safeChunkSeconds,
          maxChunkSeconds:
            workingInspection.sizeBytes > MAX_AUDIO_BYTES
              ? safeChunkSeconds
              : safeChunkSeconds + boundaryWindowSeconds,
          boundaryWindowSeconds,
          boundaries: silenceBoundaries,
          forceChunk: Boolean(opts.alwaysChunk),
        });
        chunkPaths = await extractAudioChunks({
          sourcePath: workingPath,
          ranges: chunkRanges,
          outDir: outputsDir,
          sampleRate,
          bitrate,
        });
      }

      const chunkResults: BatchChunkResult[] = [];
      let previousTranscript = "";

      for (const [index, chunkPath] of chunkPaths.entries()) {
        const chunkInspection = await inspectAudio(chunkPath);
        const range = chunkRanges[index] || {
          index,
          startSeconds: 0,
          endSeconds: chunkInspection.durationSeconds || 0,
        };
        const result = await provider.transcribe({ config }, {
          ...audioOptions,
          audioPath: chunkPath,
          prompt: buildCarryoverPrompt({
            basePrompt: audioOptions.prompt,
            previousTranscript: promptCarryover ? previousTranscript : undefined,
            model: audioOptions.model!,
          }),
        } as AudioCommandOptions);
        const outputPath = await writeTranscriptOutput({
          audioPath: chunkPath,
          responseFormat: result.responseFormat,
          outDir: chunkOutputsDir,
          content: stringifyTranscriptData(result),
        });
        chunkResults.push({
          index,
          path: chunkPath,
          offsetSeconds: range.startSeconds,
          durationSeconds: chunkInspection.durationSeconds,
          result,
          outputPath,
        });
        previousTranscript = `${previousTranscript}\n${result.transcriptText}`.trim();
      }

      const merged = mergeBatchResults(chunkResults, audioOptions.responseFormat!);
      const mergedPath = join(
        jobDir,
        `merged.transcript.${audioOptions.responseFormat === "text" ? "txt" : "json"}`,
      );
      const manifestPath = join(jobDir, "job.json");

      await writeFile(
        mergedPath,
        audioOptions.responseFormat === "text"
          ? `${String(merged.text || "")}\n`
          : `${JSON.stringify(merged, null, 2)}\n`,
        "utf8",
      );
      await writeFile(
        manifestPath,
        `${JSON.stringify(
          {
            provider: providerName,
            source: sourceInspection,
            working: workingInspection,
            requested_chunk_seconds: requestedChunkSeconds,
            boundary_window_seconds: boundaryWindowSeconds,
            min_silence_seconds: minSilenceSeconds,
            silence_threshold_db: silenceThresholdDb,
            prompt_carryover: promptCarryover,
            response_format: audioOptions.responseFormat,
            model: audioOptions.model,
            chunks: chunkResults.map((chunk) => ({
              index: chunk.index,
              path: chunk.path,
              offset_seconds: chunk.offsetSeconds,
              duration_seconds: chunk.durationSeconds,
              output_path: chunk.outputPath,
            })),
            merged_output_path: mergedPath,
          },
          null,
          2,
        )}\n`,
        "utf8",
      );

      batchOutputs.push({
        audioPath,
        jobDir,
        chunkCount: chunkResults.length,
        mergedPath,
        manifestPath,
      });
    }

    const payload = successEnvelope("transcribe.batch", {
      provider: providerName,
      outputs: batchOutputs,
    });

    if (json) {
      printJson(payload);
      return;
    }

    printHuman(
      "Batch complete",
      batchOutputs.map((item) =>
        item.dryRun ? `${item.audioPath}: dry-run` : `${item.audioPath} -> ${item.mergedPath}`,
      ),
    );
  }, command);
}

async function main(): Promise<void> {
  const program = new Command();
  program
    .name("transcribe-cli")
    .description("Provider-oriented audio transcription CLI. OpenAI ships in v1.")
    .version(VERSION)
    .option("--json", "Emit machine-readable JSON output")
    .option("--config <path>", "Use a specific config file")
    .option("--api-key <value>", "API key override for one-off tests")
    .addOption(new Option("--provider <name>", "Provider").choices(["openai"]));

  program
    .command("init")
    .description("Create a starter config file")
    .option("--force", "Overwrite an existing config file")
    .action(async function action() {
      const opts = this.optsWithGlobals() as { config?: string; json?: boolean; force?: boolean };
      await ensureConfigDir(opts.config);
      const configPath = await writeStarterConfig(opts.config, Boolean(opts.force));
      const payload = successEnvelope("init", { configPath });
      if (opts.json) {
        printJson(payload);
        return;
      }
      printHuman("Config created", [configPath]);
    });

  program
    .command("doctor")
    .description("Verify config, auth sources, and local helper tools")
    .action(async function action() {
      await withConfig(async ({ json, config }) => {
        const providerResults = await Promise.all(
          Object.values(providers).map((provider) => provider.doctor({ config })),
        );
        const payload = successEnvelope("doctor", {
          configPath: config.configPath,
          configExists: config.configExists,
          defaultProvider: config.defaultProvider,
          providers: providerResults,
          tools: {
            ffmpeg: commandOnPath("ffmpeg"),
            ffprobe: commandOnPath("ffprobe"),
          },
        });
        if (json) {
          printJson(payload);
          return;
        }
        renderDoctorHuman(payload.data);
      }, this);
    });

  const providersCommand = program.command("providers").description("Provider discovery");
  providersCommand
    .command("list")
    .description("List supported providers")
    .action(async function action() {
      await withConfig(async ({ json }) => {
        const items = Object.values(providers).map((provider) => provider.info);
        const payload = successEnvelope("providers.list", { providers: items });
        if (json) {
          printJson(payload);
          return;
        }
        printHuman(
          "Providers",
          items.map((item) => `${item.name}: ${item.title} [${item.supports.join(", ")}]`),
        );
      }, this);
    });

  program
    .command("models")
    .description("Model discovery")
    .command("list")
    .description("List built-in model metadata")
    .addOption(new Option("--provider <name>", "Provider").choices(["openai"]))
    .action(async function action() {
      await withConfig(async ({ json, config }) => {
        const providerName = selectProvider(this, config);
        const models = requireProvider(providerName).listModels();
        const payload = successEnvelope("models.list", {
          provider: providerName,
          models,
        });
        if (json) {
          printJson(payload);
          return;
        }
        printHuman(
          "Models",
          models.map(
            (model) =>
              `${model.id}: modes=${model.modes.join(",")} formats=${model.responseFormats.join(",")}`,
          ),
        );
      }, this);
    });

  program
    .command("formats")
    .description("List supported input and output formats")
    .addOption(new Option("--provider <name>", "Provider").choices(["openai"]))
    .option("--model <id>", "Filter to one model")
    .action(async function action() {
      await withConfig(async ({ json, config }) => {
        const providerName = selectProvider(this, config);
        const opts = this.optsWithGlobals() as { model?: string };
        const formats = requireProvider(providerName).listFormats();
        const filtered = opts.model
          ? {
              ...formats,
              models: formats.models.filter((model) => model.id === opts.model),
            }
          : formats;
        const payload = successEnvelope("formats", filtered);
        if (json) {
          printJson(payload);
          return;
        }
        printHuman("Formats", [
          `provider: ${providerName}`,
          `input: ${filtered.inputFormats.join(", ")}`,
          ...filtered.models.map((model) => `${model.id}: ${model.responseFormats.join(", ")}`),
          ...filtered.notes,
        ]);
      }, this);
    });

  function addAudioCommonOptions(command: Command): Command {
    return command
      .addOption(new Option("--provider <name>", "Provider").choices(["openai"]))
      .option("--model <id>", "Provider model id")
      .addOption(
        new Option("--response-format <format>", "API response format").choices(
          ALL_RESPONSE_FORMATS as unknown as string[],
        ),
      )
      .option("--language <code>", "Language hint, for example en")
      .option("--prompt <text>", "Prompt or context where supported")
      .option("--temperature <number>", "Sampling temperature", (value) =>
        parseNumberOption("temperature", value),
      )
      .option("--out <path>", "Output file path for a single input")
      .option("--out-dir <path>", "Output directory")
      .option("--stdout", "Write the transcript to stdout")
      .option("--dry-run", "Validate options and print the request plan");
  }

  addAudioCommonOptions(
    program
      .command("transcribe")
      .description("Transcribe one or more audio files")
      .argument("<audio...>", "Audio file path(s)")
      .option("--diarize", "Use the provider default diarization model")
      .option("--include-logprobs", "Include token logprobs where supported")
      .option("--timestamp <granularity>", "Timestamp granularity (repeatable)", collect, [])
      .option("--chunking-strategy <value>", "Chunking strategy for diarization models")
      .option("--known-speaker <name=path>", "Known speaker reference", collect, [])
      .action(async function action(audioPaths: string[]) {
        await runAudioCommand({
          command: this,
          audioPaths: audioPaths.map((path) => resolve(expandHome(path))),
          mode: "transcribe",
        });
      }),
  );

  addAudioCommonOptions(
    program
      .command("translate")
      .description("Translate audio into English text")
      .argument("<audio...>", "Audio file path(s)")
      .action(async function action(audioPaths: string[]) {
        await runAudioCommand({
          command: this,
          audioPaths: audioPaths.map((path) => resolve(expandHome(path))),
          mode: "translate",
        });
      }),
  );

  addAudioCommonOptions(
    program
      .command("batch")
      .description("Long-audio workflow with optional preprocessing and chunk merging")
      .argument("<audio...>", "Audio file path(s)")
      .option("--diarize", "Use the provider default diarization model")
      .option("--include-logprobs", "Include token logprobs where supported")
      .option("--timestamp <granularity>", "Timestamp granularity (repeatable)", collect, [])
      .option("--chunking-strategy <value>", "Chunking strategy for diarization models")
      .option("--known-speaker <name=path>", "Known speaker reference", collect, [])
      .option("--working-dir <path>", "Batch working directory")
      .option("--chunk-seconds <value>", "Chunk duration in seconds", (value) =>
        parseNumberOption("chunk-seconds", value),
      )
      .option("--always-chunk", "Split the working copy even when it is already under 25 MB")
      .option(
        "--boundary-window-seconds <value>",
        "How far around the target chunk length to look for silence",
        (value) => parseNumberOption("boundary-window-seconds", value),
      )
      .option(
        "--min-silence-seconds <value>",
        "Minimum detected silence length for chunk boundaries",
        (value) => parseNumberOption("min-silence-seconds", value),
      )
      .option(
        "--silence-threshold-db <value>",
        "Silence threshold in dB for ffmpeg silencedetect",
        (value) => parseNumberOption("silence-threshold-db", value),
      )
      .option("--no-prompt-carryover", "Disable previous-segment transcript carryover prompts")
      .option("--bitrate <value>", "Speech-friendly MP3 bitrate when preprocessing")
      .option("--sample-rate <value>", "Speech-friendly sample rate when preprocessing", (value) =>
        parseNumberOption("sample-rate", value),
      )
      .action(async function action(audioPaths: string[]) {
        const normalised = audioPaths.map((path) => resolve(expandHome(path)));
        await runBatchCommand(this, normalised);
      }),
  );

  program
    .command("request")
    .description("Read-only raw request escape hatch")
    .argument("<path>", "API path, for example /models")
    .addOption(
      new Option("--method <value>", "HTTP method").choices(["GET", "HEAD"]).default("GET"),
    )
    .option("--query <key=value>", "Query string pair", collect, [])
    .action(async function action(path: string) {
      await withConfig(async ({ json, config }) => {
        const opts = this.optsWithGlobals() as {
          method: "GET" | "HEAD";
          query?: string[];
        };
        const providerName = selectProvider(this, config);
        const result = await requireProvider(providerName).request(
          { config },
          {
            method: opts.method,
            path,
            query: collectQueryPairs(opts.query || []),
          },
        );
        const payload = successEnvelope("request", result);
        if (json) {
          printJson(payload);
          return;
        }
        printHuman("Request", [`${result.method} ${result.url}`, `status: ${result.status}`]);
      }, this);
    });

  await program.parseAsync(process.argv);
}

main().catch((error: unknown) => {
  if (getJsonFlag()) {
    printJson(toErrorPayload(error));
  } else if (error instanceof AppError) {
    process.stderr.write(`Error: ${error.message}\n`);
  } else if (error instanceof Error) {
    process.stderr.write(`Error: ${error.message}\n`);
  } else {
    process.stderr.write("Error: An unexpected error occurred.\n");
  }
  process.exit(error instanceof AppError ? error.exitCode : 1);
});
