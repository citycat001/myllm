import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model: a bigram model.

    It predicts the next token using ONLY the current token.
    The entire "knowledge" lives in a single embedding table where
    each row gives the logits (unnormalized probabilities) for what
    comes next after that character.

    This is intentionally minimal — no attention, no context window,
    no positional encoding. It's the baseline we'll improve on later.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # Each token directly looks up a row of logits for the next token.
        # Think of it as a (vocab_size x vocab_size) probability table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Args:
            idx:     (B, T) tensor of token indices
            targets: (B, T) tensor of target indices, or None for inference

        Returns:
            logits: (B, T, vocab_size) — raw predictions for each position
            loss:   scalar cross-entropy loss, or None if targets not provided
        """
        # idx shape: (B, T)
        # logits shape: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            # Reshape for cross_entropy: it expects (N, C) and (N,)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively generate new tokens.

        Args:
            idx:            (B, T) tensor of starting context token indices
            max_new_tokens: how many new tokens to generate

        Returns:
            (B, T + max_new_tokens) tensor with the generated continuation
        """
        for _ in range(max_new_tokens):
            # Get predictions (we only need the last time step)
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx
