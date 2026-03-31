"""
tinyllama_runner.py
===================
Loads TinyLlama-1.1B-Chat and exposes a ``generate()`` function that follows
the model's official chat template.

Usage
-----
    from tinyllama_runner import TinyLlamaRunner
    runner = TinyLlamaRunner()
    answer = runner.generate(
        system_prompt="You are a helpful assistant.",
        context="Alice said she likes apples.",
        question="What does Alice like?"
    )
"""

import logging

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


class TinyLlamaRunner:
    """Wrapper around TinyLlama-1.1B-Chat for context-window QA evaluation.

    Loads the model once at construction time and exposes a ``generate``
    method that formats messages using the official TinyLlama chat template.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful, respectful and honest assistant. "
        "Answer the question using only the information in the provided context."
    )

    def __init__(self, config_path: str = "config.yaml") -> None:
        cfg = load_config(config_path)
        tc = cfg["tinyllama"]

        self.model_name: str = tc["model_name"]
        self.max_new_tokens: int = tc["max_new_tokens"]
        self.temperature: float = float(tc.get("temperature", 0.7))
        self.do_sample: bool = bool(tc.get("do_sample", True))

        dtype = torch.float16 if tc["dtype"] == "float16" else torch.float32

        logger.info("Loading tokenizer for %s ...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info(
            "Loading model %s (dtype=%s, device_map=%s) ...",
            self.model_name, tc["dtype"], tc["device_map"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=tc["device_map"],
        )
        self.model.eval()

        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=dtype,
            device_map=tc["device_map"],
        )
        logger.info("TinyLlama model ready.")

    def _build_prompt(
        self,
        system_prompt: str,
        context: str,
        question: str,
    ) -> str:
        """Build a prompt string using TinyLlama's official chat template.

        Follows the format documented at:
        https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

        The user message contains the context followed by the question so
        the model attends to the provided material before answering.

        Parameters
        ----------
        system_prompt : str
            High-level instruction for the assistant.
        context : str
            Conversation history or compressed context to ground the answer.
        question : str
            The comprehension question to answer.

        Returns
        -------
        str
            Fully formatted prompt string ready for the model.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        prompt: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an answer to ``question`` given the provided ``context``.

        Parameters
        ----------
        system_prompt : str
            High-level instruction for the assistant role.
        context : str
            The context (conversation history or compressed summary) to
            condition the answer on.
        question : str
            The question to answer.
        max_new_tokens : int
            Maximum number of new tokens to generate.

        Returns
        -------
        str
            The model's generated answer (decoded, stripped of the prompt).
        """
        if not system_prompt:
            system_prompt = self._DEFAULT_SYSTEM

        prompt = self._build_prompt(system_prompt, context, question)

        outputs = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Strip the prompt from the generated text
        full_text: str = outputs[0]["generated_text"]
        answer = full_text[len(prompt):].strip()
        return answer


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = TinyLlamaRunner()
    demo_context = (
        "Human: What is the capital of France?\n"
        "GPT: The capital of France is Paris. It is also the country's "
        "largest city and a global centre for art, culture and fashion."
    )
    demo_question = "What is the capital of France?"
    answer = runner.generate(
        system_prompt=TinyLlamaRunner._DEFAULT_SYSTEM,
        context=demo_context,
        question=demo_question,
    )
    print("\n-- TinyLlama smoke-test --")
    print(f"Q: {demo_question}")
    print(f"A: {answer}")
