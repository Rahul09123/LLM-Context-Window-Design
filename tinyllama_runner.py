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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        logger.info("Loading tokenizer for %s ...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model %s in 4-bit NF4 ...", self.model_name)
        from transformers import BitsAndBytesConfig
        _bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=_bnb_cfg,
            device_map=tc.get("device_map", "auto"),
            torch_dtype=torch.float16,
        )
        self.model.eval()

        # _pipe is built lazily via _get_pipe() so that run_phase2 can
        # inject a pre-loaded model/tokenizer before the pipeline is used.
        self._pipe = None
        logger.info("TinyLlama model ready.")

    def _get_pipe(self):
        """Return the HuggingFace pipeline, building it once on first call.

        Lazy construction means that if run_phase2 injects a different model
        or tokenizer into self.model / self.tokenizer after __init__, the
        pipeline will be built from the injected versions, not the originals.
        """
        if self._pipe is None:
            from transformers import pipeline as _pipeline
            self._pipe = _pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        return self._pipe

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

        # Use instance max_new_tokens (set from config / overridden externally)
        # capped at the argument value. Greedy decoding saves memory vs sampling.
        _tokens = min(max_new_tokens, self.max_new_tokens)
        _do_sample = self.do_sample and self.temperature > 0.0

        pipe = self._get_pipe()
        gen_kwargs = dict(
            max_new_tokens=_tokens,
            do_sample=_do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if _do_sample:
            gen_kwargs["temperature"] = self.temperature

        import gc
        try:
            outputs = pipe(prompt, **gen_kwargs)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        # Strip the prompt from the generated text
        full_text: str = outputs[0]["generated_text"]
        answer = full_text[len(prompt):].strip()
        return answer

    def generate_batch(
        self,
        items: list[tuple[str, str, str]],
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Batched inference for multiple (system_prompt, context, question) tuples.

        Sends all prompts in a single forward pass so the GPU processes a real
        batch instead of one sample at a time.  Uses left-padding, which is
        required for decoder-only models like TinyLlama.

        Parameters
        ----------
        items : list of (system_prompt, context, question)
        max_new_tokens : int

        Returns
        -------
        list[str]  answers in the same order as ``items``
        """
        if not items:
            return []

        # Decoder-only models need left-padding for batched generation
        self.tokenizer.padding_side = "left"

        prompts = [
            self._build_prompt(sys_p or self._DEFAULT_SYSTEM, ctx, q)
            for sys_p, ctx, q in items
        ]

        _tokens = min(max_new_tokens, self.max_new_tokens)
        _do_sample = self.do_sample and self.temperature > 0.0

        pipe = self._get_pipe()
        gen_kwargs = dict(
            max_new_tokens=_tokens,
            do_sample=_do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            # batch_size here controls pipeline chunking, not model batch dim
            batch_size=min(len(prompts), 4),
        )
        if _do_sample:
            gen_kwargs["temperature"] = self.temperature

        import gc
        try:
            outputs = pipe(prompts, **gen_kwargs)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return [
            out[0]["generated_text"][len(prompt):].strip()
            for prompt, out in zip(prompts, outputs)
        ]


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