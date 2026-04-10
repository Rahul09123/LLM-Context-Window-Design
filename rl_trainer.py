"""
rl_trainer.py
=============
REINFORCE trainer for the Phase-2 context compression policy.

Uses RLAIF scores (0-10 from Claude) as the reward signal to optimise the
ContextCompressor turn-selection policy via the REINFORCE policy-gradient
algorithm with an exponential-moving-average reward baseline.

Algorithm summary
-----------------
For each batch of N episodes:
  1. Sample a random (conversation, question, gold_answer) triple.
  2. Call compressor.sample_compress() to obtain a stochastic selection
     and its log-probability under the current policy.
  3. Generate an answer with TinyLlama on the compressed context.
  4. Score the answer with Claude (RLAIF) → reward r ∈ [0, 10].
  5. Accumulate loss  = -(r - baseline) × log_prob  across the batch.
  6. Step the Adam optimiser; update the running baseline.

Usage
-----
    python rl_trainer.py
    python rl_trainer.py --config my.yaml --no-wandb --episodes 100
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
from gensim.models import Word2Vec

from context_compressor import ContextCompressor
from llm_client import build_llm_client, LLMClient
from tinyllama_runner import TinyLlamaRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RLAIF reward
# ─────────────────────────────────────────────────────────────────────────────

_RLAIF_SYSTEM = (
    "You are an objective QA evaluator. "
    "Score the answer strictly on a scale of 0 to 10 using the rubric below. "
    "Return a single JSON object with key 'score' (float). No other text."
)

_RLAIF_RUBRIC = """Rubric:
10 — Fully correct, concise, grounded in the context.
7-9 — Mostly correct with minor omissions or imprecision.
4-6 — Partially correct; misses key information.
1-3 — Largely incorrect but tangentially related.
0   — Completely wrong or refuses to answer.

Question: {question}
Gold Answer: {gold_answer}
Model Answer: {model_answer}
"""


def _rlaif_score(
    llm: LLMClient,
    question: str,
    gold_answer: str,
    model_answer: str,
) -> float:
    """Return a 0-10 RLAIF score via the configured LLM client."""
    import re as _re, json as _json

    user_msg = _RLAIF_RUBRIC.format(
        question=question, gold_answer=gold_answer, model_answer=model_answer,
    )
    raw = llm.complete(system=_RLAIF_SYSTEM, user=user_msg, max_tokens=64)
    if not raw:
        return 0.0
    try:
        return float(max(0.0, min(10.0, _json.loads(raw)["score"])))
    except Exception:
        pass
    m = _re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
    if m:
        return float(max(0.0, min(10.0, float(m.group(1)))))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _load_training_samples(config: dict) -> list[dict]:
    """Load val_qa.json and flatten to a list of (conversation, question, answer)."""
    qa_file = config["data"]["val_qa_file"]
    if not Path(qa_file).exists():
        raise FileNotFoundError(
            f"{qa_file} not found. Run qa_generator.py first."
        )
    with open(qa_file, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    samples: list[dict] = []
    for record in records:
        conv = record.get("conversations", [])
        for qa in record.get("qa_pairs", []):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a and conv:
                samples.append({"conversation": conv, "question": q, "gold_answer": a})

    logger.info("Loaded %d training samples from %s", len(samples), qa_file)
    return samples


def _load_word2vec(config: dict) -> Word2Vec:
    model_path = config["word2vec"]["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Word2Vec model not found at {model_path}. "
            "Run word2vec_trainer.py (or run_phase1.py) first."
        )
    logger.info("Loading Word2Vec from %s ...", model_path)
    return Word2Vec.load(model_path)


_QA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based only on the "
    "provided context. Be concise and accurate."
)


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE trainer
# ─────────────────────────────────────────────────────────────────────────────

class RLTrainer:
    """Trains ContextCompressor with REINFORCE.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    runner : TinyLlamaRunner, optional
        Pre-loaded TinyLlama runner (created if not provided).
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        runner: Optional[TinyLlamaRunner] = None,
    ) -> None:
        self.cfg = _load_config(config_path)
        self.p2cfg = self.cfg["phase2"]

        self._llm: LLMClient = build_llm_client(self.cfg["llm"])

        w2v = _load_word2vec(self.cfg)
        self.compressor = ContextCompressor(
            w2v_model=w2v,
            token_budget=self.cfg["harness"]["truncation_tokens"],
            hidden_dim=self.p2cfg["policy_hidden_dim"],
        )

        # Use the pre-loaded runner if provided — avoids loading a second
        # copy of TinyLlama which would cause OOM on T4 (14.56 GB VRAM).
        if runner is not None:
            self._runner = runner
        else:
            logger.warning(
                "No runner provided to RLTrainer — loading TinyLlamaRunner "
                "internally. Pass runner= from run_phase2 to avoid double load."
            )
            self._runner = TinyLlamaRunner(config_path=config_path)

        lr = float(self.p2cfg["learning_rate"])
        self.optimizer = optim.Adam(self.compressor.policy.parameters(), lr=lr)

        self.baseline: float = 5.0          # initial reward baseline (mid-scale)
        self._baseline_alpha: float = float(self.p2cfg["reward_baseline_alpha"])

    # ── Batched episode ───────────────────────────────────────────────────────

    def _run_batch(
        self, samples: list[dict], batch_size: int
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Run a full batch of RL episodes with one batched TinyLlama call.

        Steps
        -----
        1. Sample ``batch_size`` random training examples.
        2. Run ``sample_compress`` on each (CPU — fast Word2Vec + tiny MLP).
        3. Call ``generate_batch`` once so all prompts share a single GPU
           forward pass instead of ``batch_size`` separate calls.
        4. Score all answers in parallel via ``ThreadPoolExecutor`` — the
           Gemini/OpenAI API calls are network I/O and release the GIL, so
           true parallelism is achieved without multiprocessing overhead.

        Returns
        -------
        rewards   : list[float]  RLAIF scores [0, 10]
        log_probs : list[Tensor] scalar tensors attached to compute graph
        """
        batch_samples = [random.choice(samples) for _ in range(batch_size)]

        # Step 1 — stochastic context compression (CPU)
        contexts: list[str] = []
        log_probs: list[torch.Tensor] = []
        questions: list[str] = []
        gold_answers: list[str] = []

        for s in batch_samples:
            ctx, lp = self.compressor.sample_compress(s["conversation"])
            contexts.append(ctx)
            log_probs.append(lp)
            questions.append(s["question"])
            gold_answers.append(s["gold_answer"])

        # Step 2 — single batched TinyLlama forward pass (GPU)
        _max_new_tokens = getattr(self._runner, "max_new_tokens", 64)
        _max_new_tokens = min(_max_new_tokens, 64)   # cap to save VRAM
        answers = self._runner.generate_batch(
            items=[(_QA_SYSTEM_PROMPT, ctx, q) for ctx, q in zip(contexts, questions)],
            max_new_tokens=_max_new_tokens,
        )

        # Step 3 — parallel RLAIF scoring (network I/O — releases GIL)
        def _score(idx: int) -> tuple[int, float]:
            r = _rlaif_score(self._llm, questions[idx], gold_answers[idx], answers[idx])
            time.sleep(0.3)   # gentle per-thread rate-limit buffer
            return idx, r

        rewards: list[float] = [0.0] * batch_size
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for future in as_completed(pool.submit(_score, i) for i in range(batch_size)):
                idx, r = future.result()
                rewards[idx] = r

        return rewards, log_probs

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        samples: list[dict],
        total_episodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_wandb: bool = True,
    ) -> ContextCompressor:
        """Run the REINFORCE training loop.

        Parameters
        ----------
        samples : list[dict]
            Flattened list of {conversation, question, gold_answer} dicts.
        total_episodes : int, optional
            Total number of RL episodes.  Defaults to phase2.rl_episodes.
        batch_size : int, optional
            Episodes per gradient update.  Defaults to phase2.batch_size.
        use_wandb : bool
            Whether to log metrics to W&B.

        Returns
        -------
        ContextCompressor
            The trained compressor.
        """
        if total_episodes is None:
            total_episodes = int(self.p2cfg["rl_episodes"])
        if batch_size is None:
            batch_size = int(self.p2cfg["batch_size"])

        policy_path = self.p2cfg["policy_path"]
        best_reward = -float("inf")
        episode_rewards: list[float] = []

        logger.info(
            "Starting REINFORCE training — episodes=%d, batch=%d",
            total_episodes, batch_size,
        )

        for ep_start in range(0, total_episodes, batch_size):
            batch_rewards, batch_log_probs = self._run_batch(samples, batch_size)
            mean_reward = float(np.mean(batch_rewards))
            episode_rewards.extend(batch_rewards)

            # REINFORCE loss over batch
            # Build loss directly from log_prob tensors so the compute
            # graph is intact. torch.tensor(0.0) detaches from the graph.
            self.optimizer.zero_grad()
            loss_terms = [
                -(( r - self.baseline) * lp) / batch_size
                for r, lp in zip(batch_rewards, batch_log_probs)
            ]
            loss = torch.stack(loss_terms).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.compressor.policy.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            # Flush GPU after each batch — gradient buffers and optimizer
            # states otherwise accumulate in the reserved pool across episodes.
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()

            # Update EMA baseline
            self.baseline = (
                (1 - self._baseline_alpha) * self.baseline
                + self._baseline_alpha * mean_reward
            )

            ep_num = ep_start + batch_size
            logger.info(
                "Episode %4d/%d | batch_reward=%.3f | baseline=%.3f | loss=%.4f",
                ep_num, total_episodes, mean_reward, self.baseline, loss.item(),
            )

            if use_wandb:
                wandb.log({
                    "rl/episode": ep_num,
                    "rl/batch_mean_reward": mean_reward,
                    "rl/baseline": self.baseline,
                    "rl/loss": loss.item(),
                })

            # Save best checkpoint
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.compressor.save(policy_path)
                logger.info("  ✓ New best reward %.3f — saved policy → %s", best_reward, policy_path)

        # Final save regardless
        self.compressor.save(policy_path)
        logger.info(
            "Training complete. Best reward=%.3f. Policy saved → %s",
            best_reward, policy_path,
        )
        return self.compressor


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run REINFORCE training for Phase 2.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override phase2.rl_episodes in config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config)
    use_wandb = not args.no_wandb

    if use_wandb:
        wc = cfg["wandb"]
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(
            project=wc["project"],
            entity=wc.get("entity"),
            name=cfg["phase2"].get("run_name", "phase2-rl"),
            config={
                "phase": 2,
                "llm_provider": cfg["llm"]["provider"],
                "llm_model": cfg["llm"]["model"],
                "rl_episodes": cfg["phase2"]["rl_episodes"],
                "batch_size": cfg["phase2"]["batch_size"],
                "learning_rate": cfg["phase2"]["learning_rate"],
            },
        )

    trainer = RLTrainer(config_path=args.config)
    samples = _load_training_samples(cfg)

    trainer.train(
        samples=samples,
        total_episodes=args.episodes,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()