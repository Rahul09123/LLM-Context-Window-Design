"""
qa_generator.py
===============
For each conversation in the val / test splits, calls the Claude API to
generate 3 comprehension questions with gold answers, then saves the
enriched data as JSON.

Usage
-----
    python qa_generator.py                        # processes val + test
    python qa_generator.py --split val            # val only
    python qa_generator.py --config my.yaml       # custom config
"""

import json
import logging
import os
import time
import argparse
import re
from pathlib import Path
from typing import Any

import anthropic
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = (
    "You are an expert at reading conversations and generating comprehension "
    "questions. Always respond with valid JSON only — no markdown fences, no "
    "commentary outside the JSON object."
)

_QA_USER_TEMPLATE = """Below is a conversation. Generate exactly {n} comprehension
questions that test understanding of the specific information exchanged.
For each question provide a concise gold answer (1–3 sentences) grounded
in the conversation text.

Return a JSON object with this exact structure:
{{
  "qa_pairs": [
    {{"question": "...", "answer": "..."}},
    ...
  ]
}}

CONVERSATION:
{conversation}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file.

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


def _conversation_to_text(conversation: list[dict]) -> str:
    """Flatten a list of turn dicts into a readable multi-line string.

    Parameters
    ----------
    conversation : list[dict]
        Each dict has 'from' and 'value' keys.

    Returns
    -------
    str
        Human-readable conversation transcript.
    """
    lines = []
    for turn in conversation:
        role = turn.get("from", "unknown").capitalize()
        value = turn.get("value", "").strip()
        lines.append(f"{role}: {value}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from a model response string.

    Tries to parse the whole string first; falls back to a regex search for
    the first ``{ … }`` block if the response contains surrounding text.

    Parameters
    ----------
    text : str
        Raw model response.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON object is found.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No JSON object found in model response:\n{text[:500]}")


def generate_qa_for_conversation(
    client: anthropic.Anthropic,
    conversation: list[dict],
    model: str,
    max_tokens: int,
    n_questions: int,
    retry_delay: float = 2.0,
    max_retries: int = 3,
) -> list[dict[str, str]]:
    """Call Claude to generate QA pairs for a single conversation.

    Parameters
    ----------
    client : anthropic.Anthropic
        Initialised Anthropic API client.
    conversation : list[dict]
        List of turn dicts (with 'from' and 'value').
    model : str
        Claude model identifier.
    max_tokens : int
        Maximum tokens for Claude response.
    n_questions : int
        Number of QA pairs to request.
    retry_delay : float
        Seconds to wait between retries on API errors.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    list[dict]
        List of dicts with keys 'question' and 'answer'.
        Returns an empty list if generation fails after all retries.
    """
    conv_text = _conversation_to_text(conversation)
    user_msg = _QA_USER_TEMPLATE.format(n=n_questions, conversation=conv_text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_QA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
            parsed = _extract_json(raw_text)
            return parsed.get("qa_pairs", [])

        except (anthropic.APIError, anthropic.RateLimitError) as exc:
            logger.warning(
                "API error on attempt %d/%d: %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Parse error on attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(retry_delay)

    logger.error("Failed to generate QA for conversation after %d retries.", max_retries)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_qa_for_split(
    input_path: str,
    output_path: str,
    config_path: str = "config.yaml",
) -> list[dict[str, Any]]:
    """Generate QA pairs for every conversation in a data split file.

    Reads conversations from ``input_path`` (JSON list), calls Claude for each,
    and saves enriched records (original fields + 'qa_pairs') to ``output_path``.

    Parameters
    ----------
    input_path : str
        Path to the input JSON (list of conversation dicts).
    output_path : str
        Path to write the enriched JSON output.
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    list[dict]
        Enriched conversation records with 'qa_pairs' key added.
    """
    cfg = load_config(config_path)
    claude_cfg = cfg["claude"]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Loading conversations from %s …", input_path)
    with open(input_path, "r", encoding="utf-8") as fh:
        conversations = json.load(fh)
    logger.info("Loaded %d conversations.", len(conversations))

    enriched: list[dict] = []
    for i, record in enumerate(conversations):
        conv_id = record.get("id", f"idx_{i}")
        logger.info("[%d/%d] Generating QA for conversation %s …", i + 1, len(conversations), conv_id)

        qa_pairs = generate_qa_for_conversation(
            client=client,
            conversation=record.get("conversations", []),
            model=claude_cfg["model"],
            max_tokens=claude_cfg["max_tokens"],
            n_questions=claude_cfg["qa_per_conv"],
        )

        enriched.append({**record, "qa_pairs": qa_pairs})

        # light rate-limiting: ~1 req/sec to stay within free-tier limits
        time.sleep(1.0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(enriched, fh, ensure_ascii=False, indent=2)
    logger.info("✅  Saved %d enriched records → %s", len(enriched), output_path)

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA pairs via Claude.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Which split to process (default: both)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    dc = cfg["data"]

    if args.split in ("val", "both"):
        generate_qa_for_split(
            input_path=dc["val_file"],
            output_path=dc["val_qa_file"],
            config_path=args.config,
        )

    if args.split in ("test", "both"):
        generate_qa_for_split(
            input_path=dc["test_file"],
            output_path=dc["test_qa_file"],
            config_path=args.config,
        )
