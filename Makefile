SHELL := /bin/sh

.PHONY: build check test fmt lint install-local release-binaries

build:
	pnpm install
	pnpm run build

check:
	pnpm run check

test:
	pnpm run test

fmt:
	pnpm run format

lint:
	pnpm run lint

install-local: build
	mkdir -p "$(HOME)/.local/bin"
	chmod +x dist/cli.js
	ln -sf "$(PWD)/dist/cli.js" "$(HOME)/.local/bin/transcribe-cli"

release-binaries:
	chmod +x ./scripts/build-release-binaries.sh
	./scripts/build-release-binaries.sh
