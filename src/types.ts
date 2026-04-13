export type ProviderName = "openai";

export type AuthSource = "flag" | "env" | "config" | "missing";

export type TaskMode = "transcribe" | "translate";

export type OutputResponseFormat =
  | "json"
  | "text"
  | "srt"
  | "verbose_json"
  | "vtt"
  | "diarized_json";

export type TimestampGranularity = "word" | "segment";

export interface LoadedConfigFile {
  default_provider?: ProviderName;
  providers?: {
    openai?: {
      api_key?: string;
      base_url?: string;
      default_transcription_model?: string;
      default_diarization_model?: string;
      default_translation_model?: string;
    };
  };
}

export interface ProviderInfo {
  name: ProviderName;
  title: string;
  envVars: string[];
  supports: string[];
  defaultTranscriptionModel: string;
  defaultTranslationModel: string;
}

export interface ProviderModelInfo {
  id: string;
  provider: ProviderName;
  modes: TaskMode[];
  responseFormats: OutputResponseFormat[];
  supportsPrompt?: boolean;
  supportsLogprobs?: boolean;
  supportsTimestamps?: boolean;
  supportsDiarization?: boolean;
  notes?: string;
}

export interface ProviderFormatsInfo {
  provider: ProviderName;
  inputFormats: string[];
  maxUploadBytes: number;
  models: Array<{
    id: string;
    responseFormats: OutputResponseFormat[];
  }>;
  notes: string[];
}

export interface ProviderDoctor {
  provider: ProviderName;
  configured: boolean;
  authSource: AuthSource;
  baseUrl: string;
  defaultTranscriptionModel: string;
  defaultDiarizationModel: string;
  defaultTranslationModel: string;
  reachable?: boolean;
  reachabilityStatus?: number;
  reachabilityError?: string;
}

export interface OpenAIResolvedProviderConfig {
  apiKey?: string;
  authSource: AuthSource;
  baseUrl: string;
  defaultTranscriptionModel: string;
  defaultDiarizationModel: string;
  defaultTranslationModel: string;
}

export interface ResolvedConfig {
  configPath: string;
  configExists: boolean;
  defaultProvider: ProviderName;
  providers: {
    openai: OpenAIResolvedProviderConfig;
  };
}

export interface ProviderContext {
  config: ResolvedConfig;
}

export interface AudioCommandOptions {
  provider: ProviderName;
  audioPath: string;
  model?: string;
  responseFormat?: OutputResponseFormat;
  language?: string;
  prompt?: string;
  temperature?: number;
  includeLogprobs?: boolean;
  timestampGranularities?: TimestampGranularity[];
  chunkingStrategy?: string;
  knownSpeakers?: string[];
  dryRun?: boolean;
}

export interface AudioCommandResult {
  command: "transcribe" | "translate";
  provider: ProviderName;
  model: string;
  audioPath: string;
  responseFormat: OutputResponseFormat;
  transcriptText: string;
  data: unknown;
  dryRun?: boolean;
}

export interface BatchChunkResult {
  index: number;
  path: string;
  offsetSeconds: number;
  durationSeconds?: number;
  result: AudioCommandResult;
  outputPath: string;
}

export interface RawRequestOptions {
  method: "GET" | "HEAD";
  path: string;
  query: Record<string, string | string[]>;
}

export interface RawRequestResult {
  command: "request";
  provider: ProviderName;
  method: "GET" | "HEAD";
  url: string;
  status: number;
  headers: Record<string, string>;
  body: unknown;
}

export interface CommandSuccess<T> {
  ok: true;
  command: string;
  data: T;
}

export interface CommandFailure {
  ok: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}
