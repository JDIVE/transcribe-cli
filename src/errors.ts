import type { CommandFailure } from "./types.js";

function redactSecrets(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => redactSecrets(item));
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    return Object.fromEntries(
      entries.map(([key, item]) => {
        if (/(token|secret|authorization|api[_-]?key)/i.test(key)) {
          return [key, "[redacted]"];
        }
        return [key, redactSecrets(item)];
      }),
    );
  }
  if (typeof value === "string" && /^sk-[a-z0-9]/i.test(value)) {
    return "[redacted]";
  }
  return value;
}

export class AppError extends Error {
  readonly code: string;
  readonly exitCode: number;
  readonly details?: unknown;

  constructor(code: string, message: string, exitCode = 1, details?: unknown) {
    super(message);
    this.name = "AppError";
    this.code = code;
    this.exitCode = exitCode;
    this.details = details;
  }
}

export function toErrorPayload(error: unknown): CommandFailure {
  if (error instanceof AppError) {
    return {
      ok: false,
      error: {
        code: error.code,
        message: error.message,
        details: error.details ? redactSecrets(error.details) : undefined,
      },
    };
  }

  if (error instanceof Error) {
    return {
      ok: false,
      error: {
        code: "unexpected_error",
        message: error.message,
      },
    };
  }

  return {
    ok: false,
    error: {
      code: "unexpected_error",
      message: "An unexpected error occurred.",
      details: redactSecrets(error),
    },
  };
}
