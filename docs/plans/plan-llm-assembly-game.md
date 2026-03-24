---
created: 2026-03-24
status: Draft
---

# Implementation Plan: LLM Assembly Game

## Technical Context

Two-process desktop application: Godot 4.4 game frontend communicating
with a Python/FastAPI backend via localhost HTTP. The game provides
drag-and-drop aircraft assembly UI; the backend handles model training
and text generation using the existing myllm codebase.

**Architecture reference**: [architecture.md](../architecture/architecture.md)
(ADR-1: two-process, ADR-2: polling, ADR-3: static registry, ADR-4:
train.py refactor)

**Stack**: Godot 4.4 + GDScript | FastAPI + uvicorn | PyTorch (existing)
**No tech research doc exists** — stack was decided during brainstorming
and locked in architecture.

## Constitution Check

| Principle | Compliance | Notes |
|-----------|-----------|-------|
| P1 Education First | ✓ | All components prioritize learning UX |
| P2 Progressive Disclosure | ✓ | Era system drives complexity curve |
| P3 Composable Pluggable | ✓ | Existing factory pattern reused |
| P4 Honest Metaphors | ✓ | Dual labels on all components |
| P5 Runnable Output | ✓ | Assembly drives real PyTorch training |
| P6 Chinese-First | ✓ | All UI text in Chinese |
| P7 YAGNI | ✓ | v1 minimal scope, no premature features |
| P8 Single Skin First | ✓ | Aircraft only |
| P9 Sub-60s Training | ✓ | Config Loader enforces param limits |
| P10 Era = Complexity | ✓ | 5 eras in strict order |

## Design Components

### Design Component 1: Python Backend API
**ID**: DES-001
**Addresses**: REQ-004, REQ-005, REQ-006
**Wave**: 1
**Depends on**: —

FastAPI server wrapping existing myllm training and generation logic.
Exposes 4 endpoints: POST /train, GET /status, POST /train/cancel,
POST /chat, GET /health. Refactors train.py into callable
`train_model(config, progress_callback)` function. Refactors
generate.py into callable `generate_text(checkpoint, prompt, max_tokens)`
function. Adds `--config <json-path>` CLI mode to train.py for
standalone use. Training runs in background thread with cancellation
flag. Config Loader validates JSON against MODEL_CONFIGS schema and
enforces P9 parameter limits (<60s CPU training).

Known gap: MODEL_REGISTRY needs a generic "assembled" entry or
Config Loader must map era configs to existing model type names.

### Design Component 2: Component Registry (Data)
**ID**: DES-002
**Addresses**: REQ-002, REQ-003, REQ-008
**Wave**: 1
**Depends on**: —

Static GDScript data defining all game components across 5 eras.
Each component entry contains: game_name (Chinese), tech_name (English),
slot_type, era, LLM config params it controls, ability_stats dict,
and a description string. Also defines aircraft frame templates per
era (slot positions, silhouette reference, slot count). Data structure
enables filtering by era and slot type. Theme layer separated from
LLM mapping layer per architecture ADR-3.

### Design Component 3: Assembly UI
**ID**: DES-003
**Addresses**: REQ-001, REQ-003
**Wave**: 2
**Depends on**: DES-002

Godot scene with drag-and-drop interaction. Left panel shows available
components (filtered by era from DES-002). Center area shows aircraft
frame with typed slot positions. Uses Godot Control nodes
(_get_drag_data, _can_drop_data, _drop_data) for native DnD. Visual
feedback: green highlight for valid drops, red for invalid, snap
animation on placement. Components show dual labels (game + tech name)
and ability stat bars. Users can remove/swap placed components. Frame
validates completeness before enabling "试飞!" button.

### Design Component 4: Config Generator
**ID**: DES-004
**Addresses**: REQ-004
**Wave**: 2
**Depends on**: DES-002, DES-003

Converts completed assembly state into JSON matching MODEL_CONFIGS
schema. Reads placed components from Assembly UI, looks up their LLM
config params from Component Registry, merges into a single config
dict. Handles era-specific config differences (bigram has no
block_names; assembled models need embedding_type, block_names,
n_head, n_layer, dropout). Validates completeness — blocks export
if required slots are empty. Outputs JSON via HTTP Client to backend.

### Design Component 5: HTTP Client & Training Flow
**ID**: DES-005
**Addresses**: REQ-005, REQ-006
**Wave**: 2
**Depends on**: DES-001

Godot-side HTTP communication layer. Uses HTTPRequest node for async
requests. POST /train sends config JSON, receives session confirmation.
Timer node polls GET /status every 2 seconds (ADR-2). Progress bar
displays training status in game terms ("飞机正在学习飞行...",
"飞行稳定性: 72%"). Cancel button sends POST /train/cancel. On
completion, transitions to chat mode. POST /chat sends user input,
displays response in RichTextLabel with ability-level framing
(试飞学员 through 隐形战神). Backend startup check via GET /health
on game launch with "请先启动训练引擎" fallback.

### Design Component 6: Era Manager & Challenge System
**ID**: DES-006
**Addresses**: REQ-002, REQ-007
**Wave**: 3
**Depends on**: DES-002, DES-005

Manages era state machine (locked/current/completed) and challenge
evaluation. Tracks which eras are unlocked, which challenges passed.
When user generates text via chat, Era Manager captures the output and
runs challenge evaluation rules (regex + keyword list + length checks
as defined in spec). Specific rules per era:
- Era 1: contains Chinese chars; regex match 三国人名 list
- Era 2: len >= 10 + no char repeats > 3
- Era 3: >= 2 distinct names from list
- Era 4: contains quote pairs
- Era 5: len >= 50 + no repeats + has punctuation

Completing all era challenges unlocks next era and triggers save.

### Design Component 7: Session Manager
**ID**: DES-007
**Addresses**: REQ-009
**Wave**: 2
**Depends on**: DES-002

Local persistence using Godot FileAccess. Saves to user://save.json:
unlocked eras, completed challenges per era, last assembly state
(which components in which slots). Auto-saves on era completion,
challenge completion, game close. Auto-loads on startup with schema
validation — corrupt data triggers graceful reset to initial state.
No login or account required.

### Design Component 8: Chinese UI & Theming
**ID**: DES-008
**Addresses**: REQ-010, REQ-008
**Wave**: 3
**Depends on**: DES-003, DES-005, DES-006

All user-facing text strings consolidated in a Chinese string table.
Tooltips on all interactive elements. Technical terms show Chinese
explanation on first appearance (tooltip or inline). Error messages
use child-friendly language ("飞机还没装好引擎呢！" not "Error:
missing block_names"). Aircraft visual theme: top-down line drawing
style, 5 distinct era silhouettes. Ability-level labels and UI
framing for output quality per era.

## Wave Summary

| Wave | Components | What it builds |
|------|-----------|----------------|
| 1 | DES-001, DES-002 | Backend API + game data (can develop in parallel) |
| 2 | DES-003, DES-004, DES-005, DES-007 | Game UI, config export, HTTP flow, persistence |
| 3 | DES-006, DES-008 | Challenge system, Chinese UI polish |

## Requirement Coverage

| REQ ID | Requirement | Addressed By |
|--------|-------------|-------------|
| REQ-001 | Drag-and-drop assembly | DES-003 |
| REQ-002 | Era progression system | DES-002, DES-006 |
| REQ-003 | Dual-label component mapping | DES-002, DES-003 |
| REQ-004 | Config export | DES-004, DES-001 |
| REQ-005 | Training with progress | DES-001, DES-005 |
| REQ-006 | Chat with model | DES-001, DES-005 |
| REQ-007 | Challenge system | DES-006 |
| REQ-008 | Aircraft theme | DES-002, DES-008 |
| REQ-009 | Session persistence | DES-007 |
| REQ-010 | Chinese-first UI | DES-008 |

Coverage: **10/10 requirements addressed**.

## Verification Criteria

| DES-ID | "Done" looks like |
|--------|-------------------|
| DES-001 | POST /train accepts JSON config, trains model, GET /status returns progress, POST /chat returns generated text. All endpoints tested with curl/httpie. |
| DES-002 | Data files define all 5 eras with correct component-to-LLM mappings. Unit test validates each era's config produces valid MODEL_CONFIGS JSON. |
| DES-003 | Components can be dragged to slots, invalid drops rejected, assembly state visible. Manual UI testing in Godot editor. |
| DES-004 | Completed assembly exports JSON that backend /train accepts without error. Integration test: assemble → export → train succeeds. |
| DES-005 | Progress bar updates during training, cancel stops training, chat displays response. End-to-end test: assemble → train → chat. |
| DES-006 | Completing Era 1 challenges unlocks Era 2. Each challenge rule correctly evaluates sample text. Unit test per challenge rule. |
| DES-007 | Progress persists across game restart. Corrupt save.json triggers reset without crash. |
| DES-008 | All UI text is Chinese. No untranslated English-only elements. Manual inspection. |

## Must-Haves

Derived from spec success criteria — minimum for v1 release:

1. Child can complete Era 1 (biplane) without help within 15 minutes
2. All default configs train in <60 seconds on CPU (Intel i5 equivalent)
3. Drag-and-drop assembly works with visual feedback
4. Trained model produces chat responses within 5 seconds
5. Progress persists between sessions
6. All UI text in Simplified Chinese
