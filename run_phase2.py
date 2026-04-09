"""
run_phase2.py
=============
Orchestrates the full Phase-2 pipeline:

  1. Load pre-built data splits and Word2Vec model (from Phase 1).
  2. Train the RL compression policy via REINFORCE (rl_trainer.py).
  3. Load the best saved policy into ContextCompressor.
  4. Run the full harness evaluation (oracle / baseline / compressed)
     on ``phase2.eval_sample_size`` validation examples.
  5. Print a comparison summary table.
  6. Log all metrics to Weights & Biases.

Prerequisites
-------------
Run the Phase-1 pipeline first so that the following artefacts exist:
  * data/val_qa.json     (qa_generator.py)
  * models/word2vec.model (word2vec_trainer.py / run_phase1.py)

Usage
-----
    python run_phase2.py
    python run_phase2.py --config my.yaml --no-wandb
    python run_phase2.py --skip-train   # evaluate existing policy only
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
from gensim.models import Word2Vec

from context_compressor import ContextCompressor
from harness import ContextWindowHarness
from rl_trainer import RLTrainer, _load_config, _load_training_samples
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

def _compute_summary(results: list[dict]) -> dict[str, dict]:
    methods = {
        "oracle":     ("oracle_score",     1.0),
        "baseline":   ("baseline_score",   None),
        "compressed": ("compressed_score", None),
    }
    summary: dict[str, dict] = {}
    for method, (score_key, fixed_ratio) in methods.items():
        scores = [r[score_key] for r in results if score_key in r]
        if fixed_ratio is not None:
            ratios = [fixed_ratio] * len(scores)
        else:
            ratios = [r["compression_ratio"] for r in results if "compression_ratio" in r]
        summary[method] = {
            "avg_qa_score": float(np.mean(scores)) if scores else 0.0,
            "avg_compression_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "n_samples": len(scores),
        }
    return summary


def _print_summary_table(summary: dict[str, dict], phase: int = 2) -> None:
    rows = [
        {
            "method": m,
            "avg_qa_score (0-10)": f"{v['avg_qa_score']:.3f}",
            "avg_compression_ratio": f"{v['avg_compression_ratio']:.3f}",
            "n_samples": v["n_samples"],
        }
        for m, v in summary.items()
    ]
    df = pd.DataFrame(rows).sort_values("method")
    print("\n" + "=" * 62)
    print(f"  Phase-{phase} Evaluation Summary")
    print("=" * 62)
    print(df.to_string(index=False))
    print("=" * 62 + "\n")


def _load_word2vec(cfg: dict) -> Word2Vec:
    model_path = cfg["word2vec"]["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Word2Vec model not found at {model_path}. "
            "Run run_phase1.py first."
        )
    logger.info("Loading Word2Vec from %s ...", model_path)
    return Word2Vec.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    config_path: str = "config.yaml",
    use_wandb: bool = True,
    skip_train: bool = False,
) -> None:
    """Execute the full Phase-2 pipeline end-to-end.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    use_wandb : bool
        Whether to log to Weights & Biases.
    skip_train : bool
        If True, skip RL training and load the existing saved policy instead.
    """
    cfg = _load_config(config_path)
    p2cfg = cfg["phase2"]

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
            name=p2cfg.get("run_name", "phase2-rl-compressor"),
            config={
                "phase": 2,
                "rl_episodes": p2cfg["rl_episodes"],
                "batch_size": p2cfg["batch_size"],
                "learning_rate": p2cfg["learning_rate"],
                "token_budget": cfg["harness"]["truncation_tokens"],
                "policy_hidden_dim": p2cfg["policy_hidden_dim"],
                "eval_sample_size": p2cfg["eval_sample_size"],
            },
        )
        logger.info("WandB run initialised: %s", run.url if run else "N/A")

    # ── 1. Load shared artefacts ───────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1 — Loading shared artefacts (Word2Vec, data) ...")
    logger.info("=" * 50)

    w2v = _load_word2vec(cfg)
    samples = _load_training_samples(cfg)

    # ── 2. RL Training (or load existing policy) ───────────────────────
    logger.info("=" * 50)
    if skip_train:
        logger.info("STEP 2 — Loading existing RL policy (--skip-train) ...")
    else:
        logger.info("STEP 2 — Training RL compression policy ...")
    logger.info("=" * 50)

    # ── Load TinyLlama once in 4-bit — injected into runner so it does
    # not trigger a second full fp32 load inside TinyLlamaRunner.__init__
    import torch as _torch, gc as _gc
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _model_name = cfg["tinyllama"]["model_name"]
    logger.info("Loading TinyLlama in 4-bit: %s", _model_name)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        _model_name,
        quantization_config=_bnb_cfg,
        device_map="auto",
        torch_dtype=_torch.float16,
    )
    _llm_model.eval()
    _llm_tokenizer = AutoTokenizer.from_pretrained(_model_name)
    if _llm_tokenizer.pad_token is None:
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token

    runner = TinyLlamaRunner(config_path=config_path)
    # Overwrite whatever the runner loaded with our 4-bit version
    for _attr, _obj in [("model", _llm_model), ("tokenizer", _llm_tokenizer),
                         ("llm", _llm_model), ("tok", _llm_tokenizer)]:
        if hasattr(runner, _attr):
            setattr(runner, _attr, _obj)
    _gc.collect()
    _torch.cuda.empty_cache()
    logger.info("Injected 4-bit model into TinyLlamaRunner.")


    compressor = ContextCompressor(
        w2v_model=w2v,
        token_budget=cfg["harness"]["truncation_tokens"],
        hidden_dim=p2cfg["policy_hidden_dim"],
    )
    policy_path = p2cfg["policy_path"]

    if skip_train:
        if not Path(policy_path).exists():
            raise FileNotFoundError(
                f"No saved policy at {policy_path}. "
                "Run without --skip-train first."
            )
        compressor.load(policy_path)
        logger.info("Loaded policy from %s", policy_path)
    else:
        trainer = RLTrainer(config_path=config_path, runner=runner)
        trainer.compressor = compressor
        compressor = trainer.train(
            samples=samples,
            use_wandb=use_wandb,
        )

    # ── 3. Harness evaluation ──────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3 — Running Phase-2 harness evaluation ...")
    logger.info("=" * 50)

    # Aggressive flush between RL training and eval — training leaves
    # gradient buffers and optimizer states in the reserved pool.
    import torch as _torch, gc as _gc
    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
        free_mb = _torch.cuda.mem_get_info()[0] / 1e6
        logger.info("GPU cache cleared before evaluation. Free VRAM: %.0f MB", free_mb)

    # Cap generation length on the runner to reduce activation memory
    # during the 3 back-to-back generate() calls inside evaluate().
    for _attr in ("max_new_tokens", "generation_kwargs"):
        if hasattr(runner, _attr) and _attr == "max_new_tokens":
            runner.max_new_tokens = min(getattr(runner, _attr, 256), 64)
    if hasattr(runner, "generation_kwargs") and isinstance(runner.generation_kwargs, dict):
        runner.generation_kwargs["max_new_tokens"] = 64
        runner.generation_kwargs.pop("num_beams", None)
        runner.generation_kwargs["do_sample"] = False

    harness = ContextWindowHarness(
        config_path=config_path,
        runner=runner,
        compressor=compressor,
    )

    val_qa_path = cfg["data"]["val_qa_file"]
    results_path = p2cfg["results_file"]

    results = harness.run_baseline_eval(
        dataset_path=val_qa_path,
        n_samples=p2cfg["eval_sample_size"],
        output_path=results_path,
    )

    # ── 4. Summary ─────────────────────────────────────────────────────
    if results:
        summary = _compute_summary(results)
        _print_summary_table(summary, phase=2)

        if run:
            flat: dict = {}
            for method, vals in summary.items():
                for metric, value in vals.items():
                    flat[f"eval/{method}/{metric}"] = value
            wandb.log(flat)

            artifact = wandb.Artifact("phase2-rl-results", type="evaluation")
            artifact.add_file(results_path)
            run.log_artifact(artifact)
    else:
        logger.warning("No evaluation results produced. Check val_qa.json exists.")

    # ── 5. Finish ──────────────────────────────────────────────────────
    if run:
        wandb.finish()
        logger.info("WandB run finished.")

    logger.info("Phase-2 pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase-2 RL pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip RL training; load saved policy and evaluate directly",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        config_path=args.config,
        use_wandb=not args.no_wandb,
        skip_train=args.skip_train,
    )