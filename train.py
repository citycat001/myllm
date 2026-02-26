"""
Training script for the Bigram Language Model.

Run:  uv run python train.py
"""

import os
import torch
from model import BigramLanguageModel

# --------------- Hyperparameters ---------------
BATCH_SIZE = 64        # how many independent sequences to process in parallel
BLOCK_SIZE = 8         # maximum context length (for bigram, only last char matters)
MAX_STEPS = 10000      # total training iterations (more steps for larger vocab)
EVAL_INTERVAL = 1000   # how often to estimate loss
EVAL_ITERS = 200       # how many batches to average for loss estimation
LEARNING_RATE = 1e-2   # higher LR for bigram (simple model, large vocab)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------

# ---- Load data ----
data_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
with open(data_path, "r") as f:
    text = f.read()

# ---- Build character-level tokenizer ----
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}  # string to index
itos = {i: ch for i, ch in enumerate(chars)}  # index to string
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# ---- Encode the entire dataset ----
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset: {len(data):,} tokens")

# ---- Train / validation split (90% / 10%) ----
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


def get_batch(split: str):
    """Generate a small batch of inputs (x) and targets (y)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    """Average loss over EVAL_ITERS batches for both splits."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---- Create model ----
model = BigramLanguageModel(vocab_size).to(DEVICE)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")
print(f"Device: {DEVICE}")
print()

# ---- Optimizer ----
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---- Training loop ----
print("Training...")
for step in range(MAX_STEPS):

    # Evaluate periodically
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(f"  step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # Sample a batch
    xb, yb = get_batch("train")

    # Forward + backward + update
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete!")

# ---- Save model + vocab info ----
save_path = os.path.join(os.path.dirname(__file__), "bigram_model.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}, save_path)
print(f"Model saved to {save_path}")

# ---- Quick generation sample ----
print("\n--- Sample generation (300 chars) ---\n")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated = model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))
