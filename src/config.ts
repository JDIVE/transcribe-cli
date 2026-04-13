import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

import { parse } from "toml";

import { AppError } from "./errors.js";
import type { LoadedConfigFile, ProviderName, ResolvedConfig } from "./types.js";

const DEFAULTS = {
  provider: "openai" as ProviderName,
  transcriptionModel: "gpt-4o-mini-transcribe",
  diarizationModel: "gpt-4o-transcribe-diarize",
  translationModel: "whisper-1",
  baseUrl: "https://api.openai.com/v1",
};

function xdgConfigHome(): string {
  return process.env.XDG_CONFIG_HOME || join(homedir(), ".config");
}

function getConfigPath(customPath?: string): string {
  return customPath || resolve(xdgConfigHome(), "transcribe-cli", "config.toml");
}

function normaliseProvider(value: string | undefined): ProviderName | undefined {
  if (value === "openai") {
    return value;
  }
  return undefined;
}

export async function ensureConfigDir(customPath?: string): Promise<void> {
  await mkdir(dirname(getConfigPath(customPath)), { recursive: true });
}

export async function writeStarterConfig(customPath?: string, force = false): Promise<string> {
  const configPath = getConfigPath(customPath);
  if (!force && existsSync(configPath)) {
    throw new AppError("config_exists", `Config already exists at ${configPath}`);
  }
  await ensureConfigDir(customPath);
  await writeFile(
    configPath,
    [
      "# transcribe-cli configuration",
      'default_provider = "openai"',
      "",
      "[providers.openai]",
      '# api_key = "your-openai-api-key"',
      '# base_url = "https://api.openai.com/v1"',
      'default_transcription_model = "gpt-4o-mini-transcribe"',
      'default_diarization_model = "gpt-4o-transcribe-diarize"',
      'default_translation_model = "whisper-1"',
      "",
    ].join("\n"),
    "utf8",
  );
  return configPath;
}

export async function loadResolvedConfig(
  customPath?: string,
  apiKeyFlag?: string,
): Promise<ResolvedConfig> {
  const configPath = getConfigPath(customPath);
  const configExists = existsSync(configPath);
  let loaded: LoadedConfigFile = {};

  if (configExists) {
    try {
      loaded = parse(await readFile(configPath, "utf8")) as LoadedConfigFile;
    } catch (error) {
      throw new AppError("invalid_config", `Failed to parse config: ${configPath}`, 1, {
        cause: error instanceof Error ? error.message : String(error),
      });
    }
  }

  const envApiKey = process.env.OPENAI_API_KEY;
  const resolvedApiKey = apiKeyFlag || envApiKey || loaded.providers?.openai?.api_key;
  const authSource = apiKeyFlag
    ? "flag"
    : envApiKey
      ? "env"
      : loaded.providers?.openai?.api_key
        ? "config"
        : "missing";

  return {
    configPath,
    configExists,
    defaultProvider:
      normaliseProvider(process.env.TRANSCRIBE_CLI_DEFAULT_PROVIDER) ||
      loaded.default_provider ||
      DEFAULTS.provider,
    providers: {
      openai: {
        apiKey: resolvedApiKey,
        authSource,
        baseUrl:
          process.env.OPENAI_BASE_URL || loaded.providers?.openai?.base_url || DEFAULTS.baseUrl,
        defaultTranscriptionModel:
          process.env.TRANSCRIBE_CLI_OPENAI_TRANSCRIPTION_MODEL ||
          loaded.providers?.openai?.default_transcription_model ||
          DEFAULTS.transcriptionModel,
        defaultDiarizationModel:
          process.env.TRANSCRIBE_CLI_OPENAI_DIARIZATION_MODEL ||
          loaded.providers?.openai?.default_diarization_model ||
          DEFAULTS.diarizationModel,
        defaultTranslationModel:
          process.env.TRANSCRIBE_CLI_OPENAI_TRANSLATION_MODEL ||
          loaded.providers?.openai?.default_translation_model ||
          DEFAULTS.translationModel,
      },
    },
  };
}
