# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a public learning project that documents the journey of building an LLM from scratch. It serves two goals:

1. **Code** — Progressively build a language model from a simple bigram to a full transformer, step by step.
2. **Articles** — For every implementation step, produce a how-to article in `articles/` (e.g. `articles/01-baby-step-bigram-model.md`) suitable for sharing on social media. Articles should walk readers through what was built, why, and how — written as a teaching guide, not just a changelog.

When a new step is implemented, always generate a corresponding article in `articles/` with a numbered prefix matching the step order.

## Language

Articles are written in **Chinese (中文)** for publishing on WeChat Official Account (微信公众号). All communication with the user should also be in Chinese. Code and code comments remain in English.

## Project Overview

Educational character-level language model built from scratch with PyTorch. Currently implements a bigram model as a baseline, with plans to progressively add attention and transformer layers.

See [docs/setup/commands.md](docs/setup/commands.md) for build/run commands and [docs/architecture/overview.md](docs/architecture/overview.md) for architecture details.
