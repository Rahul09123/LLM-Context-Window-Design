"""
word2vec_trainer.py
===================
Trains a Word2Vec skip-gram model on the full conversation corpus and
provides a helper to embed arbitrary text as a fixed-size numpy vector.

Usage
-----
    python word2vec_trainer.py                    # trains and saves model
    python word2vec_trainer.py --config my.yaml
"""

import json
import logging
import re
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from gensim.models import Word2Vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level cache so callers can import get_turn_embedding without
# re-instantiating the model on every call.
_model_cache: Optional[Word2Vec] = None
_model_path_cache: Optional[str] = None


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


def _tokenise(text: str) -> list[str]:
    """Lowercase and split text into word tokens, removing punctuation runs.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    list[str]
        List of lowercase word tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    return tokens


def _conversations_to_sentences(conversations: list[dict]) -> list[list[str]]:
    """Convert a list of conversation records into tokenised sentences.

    Each turn value becomes one sentence (list of tokens).

    Parameters
    ----------
    conversations : list[dict]
        List of conversation records with a 'conversations' key.

    Returns
    -------
    list[list[str]]
        Tokenised sentences suitable for Word2Vec training.
    """
    sentences: list[list[str]] = []
    for record in conversations:
        for turn in record.get("conversations", []):
            text = turn.get("value", "")
            tokens = _tokenise(text)
            if tokens:
                sentences.append(tokens)
    return sentences


def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_word2vec(config_path: str = "config.yaml") -> Word2Vec:
    """Train a Word2Vec model on all conversation data and save it to disk.

    Concatenates train, val, and test splits so the vocabulary is as broad
    as possible.  Uses skip-gram (sg=1) with the parameters defined in
    ``config.yaml``.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    gensim.models.Word2Vec
        Trained model instance.
    """
    cfg = load_config(config_path)
    dc = cfg["data"]
    wc = cfg["word2vec"]

    # ── 1. Load all splits ─────────────────────────────────────────────────
    all_conversations: list[dict] = []
    for split_file in [dc["train_file"], dc["val_file"], dc["test_file"]]:
        if Path(split_file).exists():
            split_data = _load_json(split_file)
            all_conversations.extend(split_data)
            logger.info("Loaded %d conversations from %s", len(split_data), split_file)
        else:
            logger.warning("Split file not found, skipping: %s", split_file)

    if not all_conversations:
        raise FileNotFoundError(
            "No conversation data found. Run data_loader.py first."
        )

    # ── 2. Tokenise ──────────────────────────────────────────────────────
    logger.info("Tokenising %d conversations …", len(all_conversations))
    sentences = _conversations_to_sentences(all_conversations)
    logger.info("Total sentences for training: %d", len(sentences))

    # ── 3. Train ──────────────────────────────────────────────────────────
    logger.info(
        "Training Word2Vec — vector_size=%d | window=%d | min_count=%d | sg=%d | epochs=%d",
        wc["vector_size"], wc["window"], wc["min_count"], wc["sg"], wc["epochs"],
    )
    model = Word2Vec(
        sentences=sentences,
        vector_size=wc["vector_size"],
        window=wc["window"],
        min_count=wc["min_count"],
        workers=wc["workers"],
        sg=wc["sg"],
        epochs=wc["epochs"],
    )
    logger.info(
        "Training complete. Vocabulary size: %d", len(model.wv.key_to_index)
    )

    # ── 4. Save ───────────────────────────────────────────────────────────
    model_path = wc["model_path"]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info("✅  Word2Vec model saved → %s", model_path)

    # update module cache
    global _model_cache, _model_path_cache
    _model_cache = model
    _model_path_cache = model_path

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public API — embedding helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_path: str) -> Word2Vec:
    """Load (and cache) a Word2Vec model from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved Gensim Word2Vec model.

    Returns
    -------
    gensim.models.Word2Vec
    """
    global _model_cache, _model_path_cache
    if _model_cache is None or _model_path_cache != model_path:
        logger.info("Loading Word2Vec model from %s …", model_path)
        _model_cache = Word2Vec.load(model_path)
        _model_path_cache = model_path
    return _model_cache


def get_turn_embedding(
    text: str,
    model_path: str = "models/word2vec.model",
) -> np.ndarray:
    """Compute a fixed-size embedding for a conversation turn.

    Tokenises ``text``, looks up each token in the Word2Vec vocabulary, and
    returns the mean vector over all known tokens.  Returns a zero vector if
    no tokens are found in the vocabulary.

    Parameters
    ----------
    text : str
        Raw turn text (any length).
    model_path : str
        Path to the saved Word2Vec model file.

    Returns
    -------
    np.ndarray
        Shape (vector_size,) — default (200,).
    """
    model = _load_model(model_path)
    vector_size: int = model.vector_size
    tokens = _tokenise(text)

    vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv
    ]

    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Word2Vec on conversation corpus.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = train_word2vec(args.config)

    # Quick smoke test
    test_text = "What is the best way to learn machine learning?"
    emb = get_turn_embedding(test_text, model_path=load_config(args.config)["word2vec"]["model_path"])
    print(f"\n── Embedding smoke-test ───────────────────────────")
    print(f"  Input : '{test_text}'")
    print(f"  Shape : {emb.shape}")
    print(f"  Norm  : {np.linalg.norm(emb):.4f}")
    print(f"  Sample: {emb[:5]}")
    print(f"───────────────────────────────────────────────────\n")
