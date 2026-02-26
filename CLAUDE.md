# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a public learning project that documents the journey of building an LLM from scratch. It serves two goals:

1. **Code** — Progressively build a language model from a simple bigram to a full transformer, step by step.
2. **Articles** — For every implementation step, produce a how-to article in `articles/` (e.g. `articles/01-baby-step-bigram-model.md`) suitable for sharing on social media. Articles should walk readers through what was built, why, and how — written as a teaching guide, not just a changelog.

When a new step is implemented, always generate a corresponding article in `articles/` with a numbered prefix matching the step order. Before writing, read `articles/introduction.md` for the shared concept framework, roadmap, learning plan, and writing conventions.

## Language

Articles are written in **Chinese (中文)** for publishing on WeChat Official Account (微信公众号). All communication with the user should also be in Chinese. Code comments are also in Chinese, using a 通俗易懂 (accessible/plain language) style with analogies.

## Project Overview

Educational character-level language model built from scratch with PyTorch. Currently implements a bigram model as a baseline, with plans to progressively add attention and transformer layers.

See [docs/setup/commands.md](docs/setup/commands.md) for build/run commands and [docs/architecture/overview.md](docs/architecture/overview.md) for architecture details.

## Git

- Remote: `https://github.com/citycat001/myllm.git` (HTTPS)
- Push uses `gh` as git credential helper (`gh auth setup-git`). No SSH — SSH key has a passphrase and ssh-agent is not available in this shell.
- Before pushing, ensure `gh auth status` shows logged in. If not, user needs to run `gh auth login` interactively.
- Workflow: new features on branches, PR into main.
