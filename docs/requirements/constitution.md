<!--
Sync Impact Report
===================
Version change: 1.0.0 -> 1.1.0 (MINOR — 3 new principles added)
Added principles: P8 Single Skin First, P9 Sub-60s Training, P10 Era = Complexity Tier
Modified sections: Mission Statement (refined Game path description)
Removed sections: None
Templates requiring updates: None (no .specify/templates exist yet)
Follow-up TODOs: None
-->

# Project Constitution: myllm

**Version**: 1.1.0
**Ratification Date**: 2026-03-23
**Last Amended**: 2026-03-23

## Mission Statement

myllm is a public learning project that teaches how large language models
work by letting people **build one from scratch**. It combines progressive
code implementation with gamified visual assembly, so that anyone — from
children to professional developers — can intuitively understand how an
LLM is made.

The project delivers two complementary experiences:

1. **Code path** — A step-by-step Python codebase that evolves from a
   simple bigram model to a full transformer, with composable pluggable
   components.
2. **Game path** — A drag-and-drop desktop game (Godot 4.x) where users
   assemble aircraft from components that map to real LLM building blocks.
   Aircraft era evolution (biplane → stealth) drives progressive LLM
   complexity. Assembly produces JSON config → Python training script.

## Principles

### P1: Education First

Every design decision MUST prioritize learning value over engineering
elegance. If a simpler implementation teaches the concept more clearly,
it wins over a "production-grade" alternative.

**Rationale**: The project exists to teach. Complexity that doesn't serve
understanding is actively harmful.

### P2: Progressive Disclosure

Content and complexity MUST be introduced incrementally — both in the
codebase (bigram -> attention -> transformer) and in the game (simple
assemblies -> advanced configurations). No step may require knowledge
that hasn't been introduced in a prior step.

**Rationale**: Cognitive overload kills learning. Each layer builds on
the last, maintaining a smooth difficulty curve.

### P3: Composable Pluggable Architecture

All LLM components (embeddings, attention, FFN, blocks) MUST be
implemented as independent, composable plugins registered in a factory
system. New algorithms are added by registering a new plugin — never by
modifying existing ones.

**Rationale**: This directly enables the game's drag-and-drop metaphor.
Each "game piece" = one registered plugin. Open/Closed Principle also
makes the codebase easier to teach and extend.

### P4: Honest Metaphors

Game components MUST map to real LLM concepts with technical accuracy.
Visual metaphors (e.g., "engine" = attention mechanism) SHOULD be
intuitive, but MUST NOT misrepresent what the underlying algorithm does.
Each game component MUST display its real technical name alongside the
game name.

**Rationale**: The game is a teaching tool, not a toy. Misleading
metaphors create misconceptions that are harder to unlearn than
learning from scratch.

### P5: Runnable Output

Every assembly the user creates in the game MUST produce a valid,
runnable Python training script — not just a diagram or visualization.
The generated code MUST use the same composable components from the
myllm codebase.

**Rationale**: "I built this and it actually works" is the most powerful
learning moment. The game isn't a simulation; it's a real model builder.

### P6: Chinese-First, Accessible Language

All articles, UI text, code comments, and documentation MUST be written
in Chinese (Simplified) using plain, accessible language with analogies.
Technical jargon MUST be introduced with an intuitive explanation on
first use.

**Rationale**: Target audience is Chinese learners on WeChat. Accessible
language lowers barriers for younger or non-technical users.

### P7: YAGNI — No Premature Complexity

Features MUST NOT be added until they are needed for a concrete learning
step or game mechanic. No speculative abstractions, no "just in case"
infrastructure.

**Rationale**: Every unnecessary feature is a distraction from the
teaching mission and increases maintenance burden.

### P8: Single Skin First

v1 MUST ship with only one theme skin (aircraft top-down line drawings).
Additional skins MUST NOT be developed until the core game loop
(assemble → train → chat) is validated with real users.

**Rationale**: Multiple skins multiply art assets, metaphor design, and
testing effort. Validate the concept before scaling it.

### P9: Sub-60s Training

All default model configurations presented in the game MUST complete
training on CPU in under 60 seconds. Configurations that exceed this
limit MUST NOT be offered as defaults — they may be offered as optional
"advanced" presets with a clear time warning.

**Rationale**: Target audience is children. Research shows kids abandon
tasks after ~15 seconds of waiting. 60 seconds is the absolute ceiling
for a game experience.

### P10: Era = Complexity Tier

Aircraft eras MUST map to LLM complexity tiers in strict chronological
order. Users MUST NOT access a later era's components without completing
the prior era's challenges. The mapping is:
- WWI biplane → Bigram model
- WWII monoplane → + Self-Attention
- 1950s jet → + Multi-Head Attention + FFN
- Modern fighter → Mini-GPT (multi-layer Transformer)
- Stealth fighter → + Dropout, BPE, advanced components

**Rationale**: Historical aircraft evolution provides a natural,
intuitive difficulty curve that aligns with P2 (Progressive Disclosure).

## Governance

### Amendment Procedure

1. Propose the change with rationale in a PR or conversation.
2. Evaluate impact on existing principles and downstream artifacts.
3. Update this document with new version number following SemVer:
   - **MAJOR**: Principle removed, redefined, or governance changed
     incompatibly.
   - **MINOR**: New principle added or existing principle materially
     expanded.
   - **PATCH**: Wording clarifications, typo fixes, non-semantic edits.
4. Update `Last Amended` date.
5. Propagate changes to dependent templates and documentation.

### Compliance Review

All PRs that add new features or modify architecture SHOULD be checked
against these principles. Any deviation MUST be justified in the PR
description with explicit rationale for why the principle doesn't apply
in that context.
