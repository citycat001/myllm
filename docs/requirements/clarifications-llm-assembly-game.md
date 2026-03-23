# Clarification Context: LLM Assembly Game

**Session**: 2026-03-23
**Spec**: [spec-llm-assembly-game.md](spec-llm-assembly-game.md)
**Status**: Ready for planning

## Scope Boundary

Desktop drag-and-drop aircraft assembly game for children. Aircraft
components map to LLM building blocks. Assembly → JSON config → Python
training → chat → challenge tasks. 5 aircraft eras drive progressive
LLM complexity. v1 only.

## Locked Decisions

### Component-to-Aircraft Mapping
- Era-specific part mapping defined (see spec for full table)
- Parts evolve with eras: wood→metal fuselage, biplane→monoplane→swept
  wing, propeller→jet engine→twin engines
- Each era's new parts correspond to new LLM capabilities unlocked
- Mapping must be technically honest (P4) while using intuitive
  aircraft analogies

### Slot Structure
- Slot count increases per era: 3→5→7→8→9+
- v1: one algorithm per slot, using only existing myllm implementations
- No user-exposed numeric parameters in v1 (n_head, n_embd hidden)
- Differentiation via slot structure and era, not parameter tuning
- Algorithm variants (GELU, SwiGLU, RoPE, etc.) deferred to future
  versions, each paired with an article per project convention

### Challenge Evaluation
- Simple rules only: regex + keyword list (三国人名) + length checks
- No external model evaluation
- Challenges get progressively harder across eras
- Specific challenges defined per era (see spec table)

### Quality Expectation Framing
- Ability level system: 试飞学员 → 新手飞行员 → 空军中尉 →
  王牌飞行员 → 隐形战神
- UI frames output as "aircraft growth", not "model failure"
- Cross-era comparison is the primary teaching mechanism

### Carried Forward (from constitution / discovery)
- P4 Honest Metaphors: dual labels (game name + technical name)
- P8 Single Skin First: aircraft only in v1
- P9 Sub-60s Training: all default configs < 60s on CPU
- P10 Era = Complexity Tier: strict chronological progression
- Stack: Godot 4.4 + GDScript, JSON config, FastAPI IPC

## Codebase Insights

### Reusable Assets
- `MODEL_CONFIGS` dict (train.py:54-110): 6 configs already defined,
  directly mappable to JSON schema the game exports
- `build_op()` factory: creates "attention" and "ffn" ops by name
- `build_blocks()`: creates Block list from config params
- `build_embedding()`: creates embedding plugin by type name
- `MODEL_REGISTRY`: maps model type names to classes
- `CharTokenizer`: character-level tokenizer with encode/decode

### Established Patterns
- Plugin/factory pattern: new algorithms register via dict, no
  existing code modification needed (aligns with P3)
- Config-driven model construction: MODEL_CONFIGS → factory calls
  → model instance. Game just needs to produce equivalent JSON.

### Integration Points
- `train.py` needs `--config <json-path>` argument (new, small work)
- `generate.py` needs HTTP-callable wrapper (new, medium work)
- FastAPI server wrapping train + generate (new, medium work)

## Deferred Ideas

- Algorithm variant swapping (GELU, SwiGLU, MQA, GQA, RoPE, ALiBi,
  RMSNorm, Lion, Cosine LR, Warmup+Decay) — deferred to post-v1,
  each paired with a teaching article
- User-tunable numeric parameters (sliders for n_head, n_embd) —
  potentially a "B-level" difficulty mode for older students
- BPE tokenizer implementation — needed for Era 5, currently missing
  from codebase (README roadmap step 5)
