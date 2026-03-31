"""
run_phase1.py
=============
Orchestrates the full Phase-1 pipeline:

  1. Load / prepare data (data_loader)
  2. Train Word2Vec (word2vec_trainer)
  3. Run baseline evaluation on 100 val examples (harness)
  4. Print a summary table
  5. Log all metrics to Weights & Biases

Usage
-----
    python run_phase1.py
    python run_phase1.py --config my.yaml --no-wandb
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import yaml

# Local modules
from data_loader import load_and_prepare
from word2vec_trainer import train_word2vec
from harness import ContextWindowHarness
from tinyllama_runner import TinyLlamaRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.

    Returns
    -------
    dict
    """
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _compute_summary(results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute per-method average QA score and compression ratio.

    Parameters
    ----------
    results : list[dict]
        Raw result records from ContextWindowHarness.run_baseline_eval.

    Returns
    -------
    dict
        Nested dict: method -> {avg_qa_score, avg_compression_ratio}.
    """
    methods = {
        "oracle":     ("oracle_score",     1.0),       # always ratio=1
        "baseline":   ("baseline_score",   None),
        "compressed": ("compressed_score", None),
    }
    summary: dict[str, dict[str, float]] = {}
    for method, (score_key, fixed_ratio) in methods.items():
        scores = [r[score_key] for r in results if score_key in r]
        ratios = (
            [fixed_ratio] * len(scores)
            if fixed_ratio is not None
            else [r["compression_ratio"] for r in results if "compression_ratio" in r]
        )
        summary[method] = {
            "avg_qa_score": float(np.mean(scores)) if scores else 0.0,
            "avg_compression_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "n_samples": len(scores),
        }
    return summary


def _print_summary_table(summary: dict[str, dict[str, float]]) -> None:
    """Pretty-print the evaluation summary table.

    Parameters
    ----------
    summary : dict
        Output of ``_compute_summary``.
    """
    df = pd.DataFrame(
        [
            {
                "method": method,
                "avg_qa_score (0-10)": f"{v['avg_qa_score']:.3f}",
                "avg_compression_ratio": f"{v['avg_compression_ratio']:.3f}",
                "n_samples": v["n_samples"],
            }
            for method, v in summary.items()
        ]
    )
    df = df.sort_values("method")
    print("\n" + "=" * 62)
    print("  Phase-1 Baseline Evaluation Summary")
    print("=" * 62)
    print(df.to_string(index=False))
    print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config_path: str = "config.yaml", use_wandb: bool = True) -> None:
    """Execute the full Phase-1 pipeline end-to-end.

    Steps
    -----
    1. Load and filter the ShareGPT dataset; save train/val/test splits.
    2. Train Word2Vec on the full corpus.
    3. Instantiate TinyLlama and ContextWindowHarness.
    4. Run baseline evaluation on ``harness.eval_sample_size`` val examples.
    5. Print a summary table.
    6. Log all metrics to Weights & Biases.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    use_wandb : bool
        Whether to log results to Weights & Biases.
    """
    cfg = load_config(config_path)

    # ── 0. WandB init ──────────────────────────────────────────────────
    run = None
    if use_wandb:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        wc = cfg["wandb"]
        run = wandb.init(
            project=wc["project"],
            entity=wc.get("entity"),
            name=wc.get("run_name", "phase1-baseline"),
            config={
                "phase": 1,
                "truncation_tokens": cfg["harness"]["truncation_tokens"],
                "eval_sample_size": cfg["harness"]["eval_sample_size"],
                "w2v_vector_size": cfg["word2vec"]["vector_size"],
                "tinyllama_model": cfg["tinyllama"]["model_name"],
                "claude_model": cfg["claude"]["model"],
            },
        )
        logger.info("WandB run initialised: %s", run.url if run else "N/A")

    # ── 1. Data loading ────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1 — Loading and preparing data ...")
    logger.info("=" * 50)
    splits = load_and_prepare(config_path)
    dc = cfg["data"]

    if run:
        wandb.log({
            "data/train_size": len(splits["train"]),
            "data/val_size":   len(splits["val"]),
            "data/test_size":  len(splits["test"]),
        })

    # ── 2. Word2Vec training ───────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 2 — Training Word2Vec ...")
    logger.info("=" * 50)
    w2v_model = train_word2vec(config_path)

    if run:
        wandb.log({"word2vec/vocab_size": len(w2v_model.wv.key_to_index)})

    # ── 3. Initialise TinyLlama and Harness ────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3 — Initialising TinyLlama and Harness ...")
    logger.info("=" * 50)
    runner = TinyLlamaRunner(config_path=config_path)
    harness = ContextWindowHarness(config_path=config_path, runner=runner)

    # ── 4. Baseline evaluation ─────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 4 — Running baseline evaluation ...")
    logger.info("=" * 50)

    # Check if val_qa.json exists; if not, warn and fall back to val.json
    val_qa_path = dc["val_qa_file"]
    if not Path(val_qa_path).exists():
        logger.warning(
            "val_qa.json not found. Run qa_generator.py first for full eval. "
            "Falling back to val.json (no QA pairs — skipping eval)."
        )
        results: list[dict] = []
    else:
        results = harness.run_baseline_eval(
            dataset_path=val_qa_path,
            n_samples=cfg["harness"]["eval_sample_size"],
            output_path=cfg["harness"]["baseline_file"],
        )

    # ── 5. Summary table ───────────────────────────────────────────────
    if results:
        summary = _compute_summary(results)
        _print_summary_table(summary)

        if run:
            flat_metrics: dict[str, float] = {}
            for method, vals in summary.items():
                for metric, value in vals.items():
                    flat_metrics[f"eval/{method}/{metric}"] = value
            wandb.log(flat_metrics)

            # Save results as WandB artifact
            artifact = wandb.Artifact("phase1-baseline-results", type="evaluation")
            artifact.add_file(cfg["harness"]["baseline_file"])
            run.log_artifact(artifact)
    else:
        logger.warning("No evaluation results produced. Check val_qa.json exists.")

    # ── 6. Finish WandB run ────────────────────────────────────────────
    if run:
        wandb.finish()
        logger.info("WandB run finished.")

    logger.info("Phase-1 pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase-1 pipeline.")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(config_path=args.config, use_wandb=not args.no_wandb)
