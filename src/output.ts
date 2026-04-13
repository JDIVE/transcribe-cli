import type { CommandSuccess } from "./types.js";

export function successEnvelope<T>(command: string, data: T): CommandSuccess<T> {
  return {
    ok: true,
    command,
    data,
  };
}

export function printJson(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

export function printHuman(title: string, lines: string[] = []): void {
  process.stdout.write(`${title}\n`);
  for (const line of lines) {
    process.stdout.write(`${line}\n`);
  }
}
