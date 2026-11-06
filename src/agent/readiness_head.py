"""Optional linear readiness head on top of the LLM's last hidden state."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReadinessHead(nn.Module):
    """Small linear probe mapping the final token hidden state to a readiness probability in [0, 1]."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute readiness probabilities in [0, 1].

        Args:
            last_hidden_state: Tensor of shape (batch_size, hidden_size) or (batch_size, seq_len, hidden_size)
        Returns:
            Tensor of shape (batch_size,) with values in [0, 1].
        """
        if last_hidden_state.dim() == 3:
            features = last_hidden_state[:, -1, :]
        else:
            features = last_hidden_state
        logits = self.proj(self.dropout(features.float())).squeeze(-1)
        return torch.sigmoid(logits)

    @staticmethod
    def compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss against cut-off ready_fraction_t targets."""
        return nn.functional.binary_cross_entropy(preds.float(), targets.float())

