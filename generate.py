"""
Generate text from a trained Bigram Language Model.

Run:  uv run python generate.py
      uv run python generate.py --prompt "ROMEO:" --length 500
"""

import argparse
import os
import torch
from model import BigramLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained bigram model")
    parser.add_argument("--model", default="bigram_model.pt", help="Path to saved model checkpoint")
    parser.add_argument("--prompt", default="", help="Starting text for generation (empty = generate from scratch)")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate")
    args = parser.parse_args()

    # ---- Load checkpoint ----
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        print("Run 'uv run python train.py' first to train the model.")
        return

    checkpoint = torch.load(model_path, weights_only=False)
    vocab_size = checkpoint["vocab_size"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    decode = lambda l: "".join([itos[i] for i in l])
    encode = lambda s: [stoi[c] for c in s]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Rebuild model and load weights ----
    model = BigramLanguageModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ---- Prepare starting context ----
    if args.prompt:
        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    else:
        # Start with a newline (token 0 is usually \n in this dataset)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # ---- Generate ----
    generated = model.generate(context, max_new_tokens=args.length)
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
