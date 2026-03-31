"""
harness.py
==========
ContextWindowHarness — evaluates how well different context representations
preserve downstream QA performance, using TinyLlama as the inference engine
and Claude (RLAIF) as the quality judge.

Three conditions are measured:
  - oracle   : full conversation as context
  - baseline : naive truncation to the last N tokens
  - compressed: placeholder for later RL-trained compression

Usage
-----
    from harness import ContextWindowHarness
    harness = ContextWindowHarness()
    result = harness.evaluate(conversation, question, ground_truth)
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic
import yaml

from tinyllama_runner import TinyLlamaRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RLAIF scoring prompt
# ─────────────────────────────────────────────────────────────────────────────

_RLAIF_SYSTEM = (
    "You are an objective QA evaluator. "
    "Score the answer strictly on a scale of 0 to 10 using the rubric below. "
    "Return a single JSON object with key 'score' (float). No other text."
)

_RLAIF_RUBRIC_TEMPLATE = """Rubric:
10 — Fully correct, concise, grounded in the context.
7-9 — Mostly correct with minor omissions or imprecision.
4-6 — Partially correct; misses key information.
1-3 — Largely incorrect but tangentially related.
0   — Completely wrong or refuses to answer.

Question: {question}
Gold Answer: {gold_answer}
Model Answer: {model_answer}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
    """
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _conversation_to_text(turns: list[dict]) -> str:
    """Flatten conversation turns into a readable string.

    Parameters
    ----------
    turns : list[dict]
        List of dicts with 'from' and 'value' keys.

    Returns
    -------
    str
    """
    lines = []
    for turn in turns:
        role = turn.get("from", "unknown").capitalize()
        value = turn.get("value", "").strip()
        lines.append(f"{role}: {value}")
    return "\n\n".join(lines)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Naively truncate text to approximately ``max_tokens`` whitespace tokens.

    This is the Phase-1 baseline compression strategy; the RL policy will
    replace it in Phase 2.

    Parameters
    ----------
    text : str
        Full context string.
    max_tokens : int
        Approximate maximum number of whitespace-split tokens to keep.
        Truncation retains the **most recent** tokens (end of text).

    Returns
    -------
    str
        Truncated context string.
    """
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[-max_tokens:])


def _extract_score(response_text: str) -> float:
    """Parse a RLAIF score from the Claude response.

    Tries JSON parsing first; falls back to a simple regex for a numeric value.

    Parameters
    ----------
    response_text : str
        Raw text from the Claude API response.

    Returns
    -------
    float
        Score in [0, 10]. Returns 0.0 on parse failure.
    """
    try:
        data = json.loads(response_text)
        return float(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()
    try:
        data = json.loads(cleaned)
        return float(data["score"])
    except Exception:
        pass

    match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response_text)
    if match:
        return float(match.group(1))

    match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', response_text)
    if match:
        score = float(match.group(1))
        if 0 <= score <= 10:
            return score

    logger.warning("Could not parse score from: %s", response_text[:200])
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ContextWindowHarness
# ─────────────────────────────────────────────────────────────────────────────

class ContextWindowHarness:
    """Evaluates context compression strategies using TinyLlama + RLAIF scoring.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    runner : TinyLlamaRunner, optional
        Pre-initialised model runner.  Created automatically if not supplied.
    """

    _SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the question based only on the "
        "provided context. Be concise and accurate."
    )

    def __init__(
        self,
        config_path: str = "config.yaml",
        runner: TinyLlamaRunner | None = None,
    ) -> None:
        self.cfg = load_config(config_path)
        self.harness_cfg = self.cfg["harness"]
        self.claude_cfg = self.cfg["claude"]
        self.truncation_tokens: int = self.harness_cfg["truncation_tokens"]

        # Claude client for RLAIF scoring
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self._claude = anthropic.Anthropic(api_key=api_key)

        # TinyLlama inference runner
        self._runner = runner or TinyLlamaRunner(config_path=config_path)

    # ── RLAIF scoring ─────────────────────────────────────────────────────

    def _rlaif_score(
        self,
        question: str,
        gold_answer: str,
        model_answer: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> float:
        """Call Claude to score a model answer against the gold answer.

        Parameters
        ----------
        question : str
            The question that was asked.
        gold_answer : str
            The reference / ground-truth answer.
        model_answer : str
            The answer generated by TinyLlama.
        max_retries : int
            Number of retry attempts on API/parse errors.
        retry_delay : float
            Seconds to wait between retries.

        Returns
        -------
        float
            Score in [0, 10].
        """
        user_msg = _RLAIF_RUBRIC_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer,
        )
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._claude.messages.create(
                    model=self.claude_cfg["model"],
                    max_tokens=64,
                    system=_RLAIF_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text
                score = _extract_score(raw)
                return max(0.0, min(10.0, score))  # clamp to [0, 10]
            except (anthropic.APIError, anthropic.RateLimitError) as exc:
                logger.warning("RLAIF API error (attempt %d): %s", attempt, exc)
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
        return 0.0

    # ── Context strategies ────────────────────────────────────────────────

    def _oracle_context(self, conversation: list[dict]) -> str:
        """Return the full conversation as context (oracle upper bound).

        Parameters
        ----------
        conversation : list[dict]
            Raw conversation turns.

        Returns
        -------
        str
        """
        return _conversation_to_text(conversation)

    def _baseline_context(self, conversation: list[dict]) -> str:
        """Return a naively truncated context (last N whitespace tokens).

        This is the Phase-1 baseline; RL replaces this in Phase 2.

        Parameters
        ----------
        conversation : list[dict]
            Raw conversation turns.

        Returns
        -------
        str
        """
        full_text = _conversation_to_text(conversation)
        return _truncate_to_tokens(full_text, self.truncation_tokens)

    def _compressed_context(self, conversation: list[dict]) -> str:
        """Return compressed context (stub — same as baseline in Phase 1).

        In Phase 2 this will call the RL-trained compression policy.

        Parameters
        ----------
        conversation : list[dict]
            Raw conversation turns.

        Returns
        -------
        str
        """
        # Phase-1 stub: identical to naive truncation.
        # Phase-2 will plug in the learnt policy here.
        return self._baseline_context(conversation)

    # ── Main evaluation entry-point ───────────────────────────────────────

    def evaluate(
        self,
        conversation: list[dict],
        question: str,
        ground_truth: str,
    ) -> dict[str, Any]:
        """Evaluate three context strategies on a single QA example.

        Parameters
        ----------
        conversation : list[dict]
            Conversation turns (list of {'from': ..., 'value': ...}).
        question : str
            The comprehension question.
        ground_truth : str
            Gold-standard answer for RLAIF scoring.

        Returns
        -------
        dict
            Keys:
            - ``oracle_score``       : RLAIF score (0-10) with full context
            - ``baseline_score``     : RLAIF score with truncated context
            - ``compressed_score``   : RLAIF score with compressed context
            - ``recovery_rate``      : compressed_score / oracle_score (or 0)
            - ``compression_ratio``  : |baseline| / |oracle| token counts
        """
        oracle_ctx = self._oracle_context(conversation)
        baseline_ctx = self._baseline_context(conversation)
        compressed_ctx = self._compressed_context(conversation)

        # ── Generate answers ────────────────────────────────────────────
        oracle_ans = self._runner.generate(
            system_prompt=self._SYSTEM_PROMPT,
            context=oracle_ctx,
            question=question,
        )
        baseline_ans = self._runner.generate(
            system_prompt=self._SYSTEM_PROMPT,
            context=baseline_ctx,
            question=question,
        )
        compressed_ans = self._runner.generate(
            system_prompt=self._SYSTEM_PROMPT,
            context=compressed_ctx,
            question=question,
        )

        # ── RLAIF scores ────────────────────────────────────────────────
        oracle_score = self._rlaif_score(question, ground_truth, oracle_ans)
        time.sleep(1.0)
        baseline_score = self._rlaif_score(question, ground_truth, baseline_ans)
        time.sleep(1.0)
        compressed_score = self._rlaif_score(question, ground_truth, compressed_ans)
        time.sleep(1.0)

        # ── Derived metrics ─────────────────────────────────────────────
        oracle_tokens = len(oracle_ctx.split())
        baseline_tokens = len(baseline_ctx.split())
        compression_ratio = (
            baseline_tokens / oracle_tokens if oracle_tokens > 0 else 0.0
        )
        recovery_rate = (
            compressed_score / oracle_score if oracle_score > 0 else 0.0
        )

        return {
            "oracle_score": oracle_score,
            "baseline_score": baseline_score,
            "compressed_score": compressed_score,
            "recovery_rate": round(recovery_rate, 4),
            "compression_ratio": round(compression_ratio, 4),
            # keep generated answers for inspection
            "oracle_answer": oracle_ans,
            "baseline_answer": baseline_ans,
            "compressed_answer": compressed_ans,
        }

    # ── Batch evaluation ──────────────────────────────────────────────────

    def run_baseline_eval(
        self,
        dataset_path: str,
        n_samples: int | None = None,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate the baseline on N samples from a QA-enriched dataset file.

        Parameters
        ----------
        dataset_path : str
            Path to a JSON file (output of qa_generator.py) containing
            conversations with 'qa_pairs' key.
        n_samples : int, optional
            Number of conversations to evaluate.  Defaults to the value in
            config.yaml (harness.eval_sample_size).
        output_path : str, optional
            Path to write the JSON results. Defaults to
            harness.baseline_file in config.yaml.

        Returns
        -------
        list[dict]
            Per-example result dicts (same structure as ``evaluate()`` plus
            conversation id, question, and ground_truth).
        """
        if n_samples is None:
            n_samples = self.harness_cfg["eval_sample_size"]
        if output_path is None:
            output_path = self.harness_cfg["baseline_file"]

        logger.info("Loading dataset from %s ...", dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        # take first n_samples that have QA pairs
        samples = [d for d in dataset if d.get("qa_pairs")][:n_samples]
        logger.info("Evaluating %d conversations ...", len(samples))

        all_results: list[dict] = []
        for idx, record in enumerate(samples):
            conv_id = record.get("id", f"idx_{idx}")
            conversation = record.get("conversations", [])
            qa_pairs = record.get("qa_pairs", [])

            for qa in qa_pairs:
                question = qa.get("question", "")
                gold = qa.get("answer", "")
                if not question or not gold:
                    continue

                logger.info(
                    "[%d/%d] Evaluating conv=%s | Q: %s ...",
                    idx + 1, len(samples), conv_id, question[:60],
                )
                result = self.evaluate(conversation, question, gold)
                result.update({
                    "conversation_id": conv_id,
                    "question": question,
                    "ground_truth": gold,
                })
                all_results.append(result)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, ensure_ascii=False, indent=2)
        logger.info("Saved %d results -> %s", len(all_results), output_path)

        return all_results
