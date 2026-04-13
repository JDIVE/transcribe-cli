import OpenAI from "openai";
import type {
  TranscriptionCreateParamsNonStreaming,
  TranslationCreateParams,
} from "openai/resources/audio";

import { INPUT_FORMATS, MAX_AUDIO_BYTES, createFileStream, fileToDataUrl } from "../audio.js";
import { AppError } from "../errors.js";
import type {
  AudioCommandOptions,
  AudioCommandResult,
  OutputResponseFormat,
  ProviderContext,
  ProviderDoctor,
  ProviderFormatsInfo,
  ProviderInfo,
  ProviderModelInfo,
  RawRequestOptions,
  RawRequestResult,
  TimestampGranularity,
} from "../types.js";
import type { ProviderAdapter } from "./index.js";

const MODELS: ProviderModelInfo[] = [
  {
    id: "gpt-4o-mini-transcribe",
    provider: "openai",
    modes: ["transcribe"],
    responseFormats: ["json", "text"],
    supportsPrompt: true,
    supportsLogprobs: true,
    notes: "Fast default transcription model.",
  },
  {
    id: "gpt-4o-transcribe",
    provider: "openai",
    modes: ["transcribe"],
    responseFormats: ["json", "text"],
    supportsPrompt: true,
    supportsLogprobs: true,
    notes: "Higher-accuracy transcription model.",
  },
  {
    id: "gpt-4o-transcribe-diarize",
    provider: "openai",
    modes: ["transcribe"],
    responseFormats: ["json", "text", "diarized_json"],
    supportsDiarization: true,
    notes: "Speaker-aware transcription with optional known-speaker references.",
  },
  {
    id: "whisper-1",
    provider: "openai",
    modes: ["transcribe", "translate"],
    responseFormats: ["json", "text", "srt", "verbose_json", "vtt"],
    supportsPrompt: true,
    supportsTimestamps: true,
    notes: "Only model that currently supports audio translation and timestamp granularities.",
  },
];

function getClient(context: ProviderContext): OpenAI {
  const provider = context.config.providers.openai;
  if (!provider.apiKey) {
    throw new AppError("missing_auth", "Missing API key for provider openai.");
  }
  return new OpenAI({
    apiKey: provider.apiKey,
    baseURL: provider.baseUrl,
  });
}

function normaliseTranscriptionModel(context: ProviderContext, model?: string): string {
  return model || context.config.providers.openai.defaultTranscriptionModel;
}

function normaliseTranslationModel(context: ProviderContext, model?: string): string {
  return model || context.config.providers.openai.defaultTranslationModel;
}

function stringFromResponse(response: unknown): string {
  if (typeof response === "string") {
    return response;
  }
  if (
    response &&
    typeof response === "object" &&
    "text" in response &&
    typeof response.text === "string"
  ) {
    return response.text;
  }
  return JSON.stringify(response);
}

function buildTimestampGranularities(
  values?: TimestampGranularity[],
): TimestampGranularity[] | undefined {
  if (!values || values.length === 0) {
    return undefined;
  }
  return Array.from(new Set(values));
}

function buildChunkingStrategy(value: string | undefined) {
  if (!value) {
    return undefined;
  }
  if (value === "auto") {
    return "auto" as const;
  }
  try {
    return JSON.parse(value) as {
      type: "server_vad";
      threshold?: number;
      prefix_padding_ms?: number;
      silence_duration_ms?: number;
    };
  } catch {
    throw new AppError(
      "invalid_chunking_strategy",
      "chunking strategy must be 'auto' or a JSON VAD object.",
    );
  }
}

async function buildKnownSpeakerPayload(values: string[] | undefined) {
  const knownSpeakers = values || [];
  if (knownSpeakers.length === 0) {
    return undefined;
  }
  if (knownSpeakers.length > 4) {
    throw new AppError("too_many_known_speakers", "Known speaker references are limited to 4.");
  }
  const names: string[] = [];
  const references: string[] = [];

  for (const value of knownSpeakers) {
    const index = value.indexOf("=");
    if (index === -1) {
      throw new AppError("invalid_known_speaker", `Expected NAME=PATH, got: ${value}`);
    }
    const name = value.slice(0, index).trim();
    const path = value.slice(index + 1).trim();
    if (!name || !path) {
      throw new AppError("invalid_known_speaker", `Expected NAME=PATH, got: ${value}`);
    }
    names.push(name);
    references.push(await fileToDataUrl(path));
  }

  return {
    known_speaker_names: names,
    known_speaker_references: references,
  };
}

async function transcribeInternal(
  context: ProviderContext,
  options: AudioCommandOptions,
): Promise<AudioCommandResult> {
  const client = getClient(context);
  const model = normaliseTranscriptionModel(context, options.model);
  const responseFormat = options.responseFormat as OutputResponseFormat;
  const timestampGranularities = buildTimestampGranularities(options.timestampGranularities);
  const knownSpeakerPayload = await buildKnownSpeakerPayload(options.knownSpeakers);
  const chunkingStrategy = buildChunkingStrategy(options.chunkingStrategy);
  const payload: TranscriptionCreateParamsNonStreaming = {
    file: createFileStream(options.audioPath),
    model,
    stream: false as const,
    response_format: responseFormat,
    language: options.language,
    prompt: options.prompt,
    temperature: options.temperature,
    include: options.includeLogprobs ? ["logprobs" as const] : undefined,
    timestamp_granularities: timestampGranularities,
    chunking_strategy: chunkingStrategy,
    known_speaker_names: knownSpeakerPayload?.known_speaker_names,
    known_speaker_references: knownSpeakerPayload?.known_speaker_references,
  };

  if (options.dryRun) {
    return {
      command: "transcribe",
      provider: "openai",
      model,
      audioPath: options.audioPath,
      responseFormat,
      transcriptText: "",
      data: {
        request: {
          ...payload,
          file: options.audioPath,
          extra_body: knownSpeakerPayload,
        },
      },
      dryRun: true,
    };
  }

  const response = await client.audio.transcriptions.create(payload);

  return {
    command: "transcribe",
    provider: "openai",
    model,
    audioPath: options.audioPath,
    responseFormat,
    transcriptText: stringFromResponse(response),
    data: response,
  };
}

async function translateInternal(
  context: ProviderContext,
  options: AudioCommandOptions,
): Promise<AudioCommandResult> {
  const client = getClient(context);
  const model = normaliseTranslationModel(context, options.model);
  const responseFormat = options.responseFormat as OutputResponseFormat;
  const payload: TranslationCreateParams = {
    file: createFileStream(options.audioPath),
    model,
    response_format: responseFormat === "diarized_json" ? "json" : responseFormat,
    prompt: options.prompt,
    temperature: options.temperature,
  };

  if (options.dryRun) {
    return {
      command: "translate",
      provider: "openai",
      model,
      audioPath: options.audioPath,
      responseFormat,
      transcriptText: "",
      data: {
        request: {
          ...payload,
          file: options.audioPath,
        },
      },
      dryRun: true,
    };
  }

  const response = await client.audio.translations.create(payload);
  return {
    command: "translate",
    provider: "openai",
    model,
    audioPath: options.audioPath,
    responseFormat,
    transcriptText: stringFromResponse(response),
    data: response,
  };
}

export class OpenAIProvider implements ProviderAdapter {
  readonly info: ProviderInfo = {
    name: "openai",
    title: "OpenAI Audio",
    envVars: ["OPENAI_API_KEY"],
    defaultTranscriptionModel: "gpt-4o-mini-transcribe",
    defaultTranslationModel: "whisper-1",
    supports: ["transcribe", "translate", "batch", "raw-request"],
  };

  listModels(): ProviderModelInfo[] {
    return MODELS;
  }

  listFormats(): ProviderFormatsInfo {
    return {
      provider: "openai",
      inputFormats: INPUT_FORMATS,
      maxUploadBytes: MAX_AUDIO_BYTES,
      models: MODELS.map((model) => ({
        id: model.id,
        responseFormats: model.responseFormats,
      })),
      notes: [
        "The file upload limit for the file transcription endpoints is 25 MB.",
        "whisper-1 is the only currently documented model for audio translation and timestamp granularities.",
        "gpt-4o-transcribe-diarize supports known speaker references and diarized_json output.",
      ],
    };
  }

  async doctor(context: ProviderContext): Promise<ProviderDoctor> {
    const provider = context.config.providers.openai;
    const result: ProviderDoctor = {
      provider: "openai",
      configured: Boolean(provider.apiKey),
      authSource: provider.authSource,
      baseUrl: provider.baseUrl,
      defaultTranscriptionModel: provider.defaultTranscriptionModel,
      defaultDiarizationModel: provider.defaultDiarizationModel,
      defaultTranslationModel: provider.defaultTranslationModel,
    };

    if (!provider.apiKey) {
      return result;
    }

    try {
      const response = await fetch(`${provider.baseUrl}/models`, {
        headers: {
          authorization: `Bearer ${provider.apiKey}`,
        },
      });
      result.reachable = response.ok;
      result.reachabilityStatus = response.status;
      if (!response.ok) {
        result.reachabilityError = response.statusText;
      }
    } catch (error) {
      result.reachable = false;
      result.reachabilityError = error instanceof Error ? error.message : String(error);
    }

    return result;
  }

  async transcribe(
    context: ProviderContext,
    options: AudioCommandOptions,
  ): Promise<AudioCommandResult> {
    return transcribeInternal(context, options);
  }

  async translate(
    context: ProviderContext,
    options: AudioCommandOptions,
  ): Promise<AudioCommandResult> {
    const model = normaliseTranslationModel(context, options.model);
    if (model !== "whisper-1") {
      throw new AppError(
        "invalid_translation_model",
        "OpenAI audio translation currently supports whisper-1 only.",
      );
    }
    return translateInternal(context, { ...options, model });
  }

  async request(context: ProviderContext, options: RawRequestOptions): Promise<RawRequestResult> {
    const provider = context.config.providers.openai;
    if (!provider.apiKey) {
      throw new AppError("missing_auth", "Missing API key for provider openai.");
    }

    if (!["GET", "HEAD"].includes(options.method)) {
      throw new AppError("invalid_method", "Only GET and HEAD are supported for raw requests.");
    }

    const path = options.path.startsWith("/") ? options.path : `/${options.path}`;
    const url = new URL(`${provider.baseUrl}${path}`);
    for (const [key, raw] of Object.entries(options.query)) {
      const values = Array.isArray(raw) ? raw : [raw];
      for (const value of values) {
        url.searchParams.append(key, value);
      }
    }

    const response = await fetch(url, {
      method: options.method,
      headers: {
        authorization: `Bearer ${provider.apiKey}`,
      },
    });

    let body: unknown = null;
    const contentType = response.headers.get("content-type") || "";
    if (options.method !== "HEAD") {
      if (contentType.includes("application/json")) {
        body = await response.json();
      } else {
        body = await response.text();
      }
    }

    const headers = Object.fromEntries(response.headers.entries());
    return {
      command: "request",
      provider: "openai",
      method: options.method,
      url: url.toString(),
      status: response.status,
      headers,
      body,
    };
  }
}
