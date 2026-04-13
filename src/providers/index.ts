import type {
  AudioCommandOptions,
  AudioCommandResult,
  ProviderContext,
  ProviderDoctor,
  ProviderFormatsInfo,
  ProviderInfo,
  ProviderModelInfo,
  RawRequestOptions,
  RawRequestResult,
} from "../types.js";

export interface ProviderAdapter {
  readonly info: ProviderInfo;
  listModels(): ProviderModelInfo[];
  listFormats(): ProviderFormatsInfo;
  doctor(context: ProviderContext): Promise<ProviderDoctor>;
  transcribe(context: ProviderContext, options: AudioCommandOptions): Promise<AudioCommandResult>;
  translate(context: ProviderContext, options: AudioCommandOptions): Promise<AudioCommandResult>;
  request(context: ProviderContext, options: RawRequestOptions): Promise<RawRequestResult>;
}
