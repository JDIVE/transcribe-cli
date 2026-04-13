# transcribe-cli

`transcribe-cli` is a provider-oriented command-line tool for audio transcription workflows.

Version `0.1.x` ships with OpenAI only, but the codebase is deliberately structured around provider adapters so other backends can be added without redesigning the CLI.

## What v1 covers

- `doctor` for config, auth, and local tool checks
- `providers list`
- `models list`
- `formats`
- `transcribe` for one or more files
- `batch` for long-audio preprocessing, chunking, and merge workflows
- `translate` for OpenAI audio-to-English translation
- `request` as a read-only raw escape hatch

## Verified OpenAI audio capabilities behind this CLI

Based on the current official OpenAI docs, this tool is built around the following documented file-audio capabilities:

- File transcription via `POST /v1/audio/transcriptions`
- Models: `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, and `whisper-1`
- Response formats:
  - `whisper-1`: `json`, `text`, `srt`, `verbose_json`, `vtt`
  - `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`: `json`, `text`
  - `gpt-4o-transcribe-diarize`: `json`, `text`, `diarized_json`
- `whisper-1` timestamp granularities via `verbose_json`
- `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` logprobs
- Known speaker references for `gpt-4o-transcribe-diarize`
- Audio translation to English via `POST /v1/audio/translations`, currently documented for `whisper-1`
- 25 MB upload limit for file transcription/translation endpoints

Not everything documented is exposed in v1. In particular, live streaming of completed-file transcription events and Realtime transcription sessions are intentionally deferred so the first release stays tight and predictable.

## Install

### GitHub release binaries

Download the binary for your platform from GitHub Releases and put it on your `PATH`.

Release assets are named like:

- `transcribe-cli-v0.1.1-linux-x64-baseline`
- `transcribe-cli-v0.1.1-linux-arm64`
- `transcribe-cli-v0.1.1-darwin-x64`
- `transcribe-cli-v0.1.1-darwin-arm64`
- `transcribe-cli-v0.1.1-windows-x64-baseline.exe`

### Local install

```bash
git clone git@github.com:JDIVE/transcribe-cli.git
cd transcribe-cli
make install-local
```

That installs a symlinked `transcribe-cli` into `~/.local/bin`.

## Requirements

- Node.js 20+ for source installs
- `OPENAI_API_KEY` for OpenAI-backed commands
- `ffmpeg` and `ffprobe` are optional but strongly recommended for long-audio workflows

## Auth and config

Auth precedence:

1. `--api-key`
2. `OPENAI_API_KEY`
3. `~/.config/transcribe-cli/config.toml`

Create a starter config:

```bash
transcribe-cli init
```

Example config:

```toml
default_provider = "openai"

[providers.openai]
default_transcription_model = "gpt-4o-mini-transcribe"
default_diarization_model = "gpt-4o-transcribe-diarize"
default_translation_model = "whisper-1"
```

## Command reference

### Health and discovery

```bash
transcribe-cli doctor
transcribe-cli --json doctor
transcribe-cli providers list
transcribe-cli models list
transcribe-cli formats
```

### Transcription

Default fast transcription:

```bash
transcribe-cli transcribe meeting.m4a
```

Structured JSON output:

```bash
transcribe-cli --json transcribe meeting.m4a --response-format json
```

Whisper timestamps:

```bash
transcribe-cli transcribe meeting.m4a \
  --model whisper-1 \
  --response-format verbose_json \
  --timestamp word \
  --timestamp segment
```

Diarized transcription:

```bash
transcribe-cli transcribe meeting.wav \
  --diarize \
  --known-speaker "Agent=agent.wav" \
  --known-speaker "Customer=customer.wav"
```

### Long audio

If the source is oversized, `batch` can create a speech-friendly working copy, split it, transcribe each chunk, and produce merged output plus a job manifest.

```bash
transcribe-cli batch long-meeting.m4a
```

Always chunk even when the file is already under the upload limit:

```bash
transcribe-cli batch long-meeting.m4a --always-chunk --chunk-seconds 600
```

### Translation

```bash
transcribe-cli translate german.mp3
```

### Raw request

`request` is intentionally read-only in v1:

```bash
transcribe-cli --json request /models
transcribe-cli --json request /models/gpt-4o-transcribe
```

## JSON policy

When `--json` is enabled:

- Success output is wrapped as:

```json
{
  "ok": true,
  "command": "doctor",
  "data": {}
}
```

- Error output is wrapped as:

```json
{
  "ok": false,
  "error": {
    "code": "missing_auth",
    "message": "Missing API key for provider openai."
  }
}
```

- Secrets are never printed in command output.
- The raw `request` command returns the provider response body inside the normal CLI success envelope.

## Provider design

The current codebase keeps provider-specific behaviour behind a small adapter surface:

- provider info
- model metadata
- format metadata
- doctor checks
- transcription
- translation
- raw read-only request handling

That keeps the user-facing command shape stable while letting future providers add their own auth, model defaults, and request builders.

## Development

```bash
pnpm install
pnpm run format:check
pnpm run lint
pnpm run check
pnpm run build
pnpm run test
make install-local
```

## Scope notes

- v1 does not attempt speaker diarization on providers that do not support it.
- v1 does not fake timestamp support on models that do not document it.
- v1 does not allow raw write methods in `request`.
