# Commands

## Training

```bash
# Train the model (saves checkpoint to bigram_model.pt)
uv run python train.py
```

## Text Generation

```bash
# Generate text from trained model
uv run python generate.py
uv run python generate.py --prompt "ROMEO:" --length 500
```

## Package Management

Package management uses `uv` (not pip). Add dependencies with `uv add <package>`.
