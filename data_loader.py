"""
data_loader.py
==============
Loads the ShareGPT_Vicuna_unfiltered dataset from HuggingFace, applies quality
filters, splits into train / val / test, and saves the results as JSON files.

Usage
-----
    python data_loader.py                   # uses config.yaml defaults
    python data_loader.py --config my.yaml  # custom config path
"""

import json
import logging
import os
import random
import argparse
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
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
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", config_path)
    return cfg


def _is_valid_turn(turn: dict) -> bool:
    """Return True if a conversation turn has a recognised role and non-empty value.

    Only 'human' and 'gpt' roles are accepted (ShareGPT convention).

    Parameters
    ----------
    turn : dict
        A single conversation turn, expected to have 'from' and 'value' keys.

    Returns
    -------
    bool
    """
    role = turn.get("from", "").lower()
    value = turn.get("value", "")
    return role in {"human", "gpt"} and isinstance(value, str) and value.strip() != ""


def filter_conversations(
    raw_dataset,
    min_turns: int = 10,
) -> list[dict[str, Any]]:
    """Filter the raw dataset to keep only high-quality conversations.

    Criteria
    --------
    * Only turns with role 'human' or 'gpt' are retained.
    * Conversations must have at least ``min_turns`` valid turns after filtering.

    Parameters
    ----------
    raw_dataset : datasets.Dataset
        The raw HuggingFace dataset object.
    min_turns : int
        Minimum number of valid turns required to keep a conversation.

    Returns
    -------
    list[dict]
        List of filtered conversation dictionaries, each with keys:
        ``id``, ``conversations`` (list of {from, value} dicts).
    """
    logger.info(
        "Filtering dataset — requiring >= %d valid turns per conversation …", min_turns
    )
    kept: list[dict] = []
    skipped = 0

    for example in raw_dataset:
        conv_id = example.get("id", "")
        raw_turns = example.get("conversations", [])

        # keep only human / gpt turns with non-empty text
        clean_turns: list[dict] = [t for t in raw_turns if _is_valid_turn(t)]

        if len(clean_turns) < min_turns:
            skipped += 1
            continue

        kept.append({"id": conv_id, "conversations": clean_turns})

    logger.info(
        "Kept %d conversations, discarded %d (< %d turns).",
        len(kept),
        skipped,
        min_turns,
    )
    return kept


def split_data(
    conversations: list[dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Randomly shuffle and split data into train / val / test sets.

    Parameters
    ----------
    conversations : list[dict]
        Full list of filtered conversations.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list, list, list]
        (train, val, test) conversation lists.
    """
    random.seed(seed)
    data = conversations.copy()
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d", len(train), len(val), len(test)
    )
    return train, val, test


def save_json(data: list[dict], path: str) -> None:
    """Serialise a list of conversation dicts to a JSON file.

    Parameters
    ----------
    data : list[dict]
        Data to serialise.
    path : str
        Destination file path (parent directories are created automatically).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d records → %s", len(data), path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(config_path: str = "config.yaml") -> dict[str, list[dict]]:
    """End-to-end data loading, filtering, splitting, and saving.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test', each mapping to the
        corresponding list of conversation dictionaries.
    """
    cfg = load_config(config_path)
    dc = cfg["data"]

    # ── 1. Download / cache from HuggingFace ──────────────────────────────
    data_files = dc.get("data_files")
    if data_files:
        logger.info("Loading dataset from explicit data_files: %s", data_files)
        raw_ds = load_dataset("json", data_files=data_files, split="train")
    else:
        # Fallback: standard split-based loading
        logger.info("Loading dataset '%s' …", dc["dataset_name"])
        raw_ds = load_dataset(dc["dataset_name"], split=dc.get("dataset_split", "train"))
    logger.info("Raw dataset size: %d examples", len(raw_ds))

    # The JSON files store conversations as a list under key 'conversations'
    # (already the correct format). However the top-level is a list of objects.
    raw = raw_ds

    # ── 2. Filter ──────────────────────────────────────────────────────────
    conversations = filter_conversations(raw, min_turns=dc["min_turns"])

    # ── 3. Split ───────────────────────────────────────────────────────────
    train, val, test = split_data(
        conversations,
        train_ratio=dc["train_ratio"],
        val_ratio=dc["val_ratio"],
        seed=dc["seed"],
    )

    # ── 4. Save ────────────────────────────────────────────────────────────
    save_json(train, dc["train_file"])
    save_json(val, dc["val_file"])
    save_json(test, dc["test_file"])

    logger.info("✅  Data preparation complete.")
    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ShareGPT data splits.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    splits = load_and_prepare(args.config)
    print("\n── Dataset summary ───────────────────────────────")
    for split_name, records in splits.items():
        print(f"  {split_name:5s}: {len(records):>6,} conversations")
    print("──────────────────────────────────────────────────\n")
