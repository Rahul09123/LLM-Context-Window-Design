"""
context_compressor.py
=====================
Phase-2 context compression policy.

Implements a learned extractive turn-selection policy (ContextCompressor)
that replaces the naive truncation used in Phase 1.  The policy uses
Word2Vec turn embeddings fed through a small MLP to produce per-turn
relevance logits; turns are then greedily selected within a token budget.

The policy is trained via REINFORCE in rl_trainer.py.

Public API
----------
    compressor = ContextCompressor(w2v_model, token_budget=512)
    # deterministic inference (after training)
    context_str = compressor.compress(conversation)
    # stochastic sampling (during REINFORCE training)
    context_str, log_prob = compressor.sample_compress(conversation)
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lowercase and extract alphanumeric word tokens."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Neural policy
# ─────────────────────────────────────────────────────────────────────────────

class TurnScoringPolicy(nn.Module):
    """Two-layer MLP: turn embedding → scalar relevance logit.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input Word2Vec turn embeddings.
    hidden_dim : int
        Width of the hidden layer.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-turn logits.

        Parameters
        ----------
        x : Tensor  shape (n_turns, embedding_dim)

        Returns
        -------
        Tensor  shape (n_turns,)
        """
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Compressor
# ─────────────────────────────────────────────────────────────────────────────

class ContextCompressor:
    """Learned extractive context compressor.

    Embeds each conversation turn with Word2Vec, scores them with a
    trained MLP, then greedily selects turns within a token budget.

    Parameters
    ----------
    w2v_model : Word2Vec
        Trained Gensim Word2Vec model.
    token_budget : int
        Maximum whitespace-split tokens in the compressed output.
    hidden_dim : int
        Hidden layer size of the scoring MLP.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        w2v_model: Word2Vec,
        token_budget: int = 512,
        hidden_dim: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.w2v = w2v_model
        self.embedding_dim: int = w2v_model.vector_size
        self.token_budget = token_budget
        self.device = torch.device(device)
        self.policy = TurnScoringPolicy(self.embedding_dim, hidden_dim).to(self.device)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_turn(self, turn: dict) -> np.ndarray:
        """Mean-pool Word2Vec vectors for all known tokens in a turn."""
        tokens = _tokenise(turn.get("value", ""))
        vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        if not vecs:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    @staticmethod
    def _turn_to_text(turn: dict) -> str:
        role = turn.get("from", "unknown").capitalize()
        value = turn.get("value", "").strip()
        return f"{role}: {value}"

    def _forward(
        self, conversation: list[dict]
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Compute policy logits and per-turn metadata.

        Returns
        -------
        logits : Tensor  shape (n_turns,)
        turn_texts : list[str]
        turn_token_costs : list[int]
            Whitespace-split token count + 1 (for the newline separator).
        """
        embeddings = np.array(
            [self._embed_turn(t) for t in conversation], dtype=np.float32
        )
        emb_t = torch.from_numpy(embeddings).to(self.device)
        logits = self.policy(emb_t)
        turn_texts = [self._turn_to_text(t) for t in conversation]
        turn_costs = [len(txt.split()) + 1 for txt in turn_texts]
        return logits, turn_texts, turn_costs

    def _apply_budget(
        self,
        score_order: list[int],
        selection_mask: list[bool],
        turn_costs: list[int],
    ) -> list[int]:
        """Greedy budget application: iterate turns by descending score.

        Parameters
        ----------
        score_order : list[int]
            Turn indices sorted by descending score.
        selection_mask : list[bool]
            Whether each turn was *sampled* as selected.
        turn_costs : list[int]
            Token cost of including each turn.

        Returns
        -------
        list[int]
            Indices of turns that fit within the budget, in score order
            (caller re-sorts to conversation order).
        """
        selected: list[int] = []
        budget = self.token_budget
        for idx in score_order:
            if selection_mask[idx] and budget >= turn_costs[idx]:
                selected.append(idx)
                budget -= turn_costs[idx]
        return selected

    # ── Public API ────────────────────────────────────────────────────────────

    def compress(self, conversation: list[dict]) -> str:
        """Deterministic greedy compression used at *inference* time.

        Selects turns in descending policy-score order until the token
        budget is exhausted, then returns them in their original order.

        Parameters
        ----------
        conversation : list[dict]
            Conversation turns, each with 'from' and 'value' keys.

        Returns
        -------
        str
            Compressed context string.
        """
        with torch.no_grad():
            logits, turn_texts, turn_costs = self._forward(conversation)

        scores = logits.cpu().numpy()
        score_order = list(np.argsort(scores)[::-1])

        selected = self._apply_budget(
            score_order,
            [True] * len(conversation),   # all turns are candidates
            turn_costs,
        )
        if not selected:
            selected = [score_order[0]]   # always keep at least one turn

        selected.sort()
        return "\n\n".join(turn_texts[i] for i in selected)

    def sample_compress(
        self, conversation: list[dict]
    ) -> tuple[str, torch.Tensor]:
        """Stochastic compression used during *REINFORCE training*.

        Each turn is included with probability sigmoid(logit).  The
        budget constraint is applied greedily over the sampled subset,
        processing turns in descending score order.

        Parameters
        ----------
        conversation : list[dict]

        Returns
        -------
        context : str
            Sampled compressed context.
        log_prob : torch.Tensor  scalar
            Sum of Bernoulli log-probabilities for the sampled decisions,
            used to compute the policy gradient.
        """
        logits, turn_texts, turn_costs = self._forward(conversation)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs)
        sampled = dist.sample()
        log_prob: torch.Tensor = dist.log_prob(sampled).sum()

        score_order = torch.argsort(logits, descending=True).tolist()
        selection_mask = [bool(sampled[i].item() > 0.5) for i in range(len(conversation))]

        selected = self._apply_budget(score_order, selection_mask, turn_costs)
        if not selected:
            selected = [score_order[0]]

        selected.sort()
        context = "\n\n".join(turn_texts[i] for i in selected)
        return context, log_prob

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save policy weights to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy weights from ``path``."""
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
