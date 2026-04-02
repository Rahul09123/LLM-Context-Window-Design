# LLM Context Window Design

This project evaluates how well different context compression and representation strategies preserve downstream Question-Answering (QA) performance. It uses TinyLlama as the inference engine and Claude as a Reinforcement Learning from AI Feedback (RLAIF) quality judge.

## 1. Current Implementation (Phase 1)

The current implementation represents Phase 1 of the project, which establishes the baseline evaluation framework and naive context processing.

Currently, the system models three context conditions:
* **Oracle:** The entire conversation history is provided as context.
* **Baseline:** Naive truncation, keeping only the last N tokens of the conversation.
* **Compressed (Placeholder):** A placeholder for future RL-trained context compression policies (currently defaults to the baseline implementation).

## 2. What Exactly is Happening in the Code

The codebase is modular and divided into specific steps of the pipeline:

* **`data_loader.py`**: Fetches the `ShareGPT_Vicuna_unfiltered` dataset from HuggingFace, filters out low-quality conversations (keeping only valid human/gpt exchanges with >= 10 turns), splits them into train/validation/test sets, and saves them locally.
* **`qa_generator.py`**: Reads the validation and test datasets and calls the Claude API to automatically generate comprehension questions and corresponding "gold" answers based directly on the conversation context.
* **`word2vec_trainer.py`**: Trains a Word2Vec skip-gram model on the entire conversation corpus. It provides a fixed-size embedding representation for conversation turns.
* **`tinyllama_runner.py`**: An inference wrapper orchestrating the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model. It accurately applies the official formatting templates and generates model responses.
* **`harness.py`**: The core evaluation framework. It runs the conversational contexts and QA tasks through TinyLlama and uses the Claude API as an RLAIF evaluator to score TinyLlama's generated answers (0-10) against the gold standard.
* **`run_phase1.py`**: The main orchestrator script that executes the entire Phase-1 pipeline end-to-end, evaluates the baseline, prints performance summaries, and continuously logs metrics to Weights & Biases (wandb).
* **`config.yaml`**: The centralized configuration file containing all variable settings like hyperparameters, sample sizes, and file paths.

## 3. How to Start the Application

### Prerequisites

1. Ensure your Python environment is set up and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Export your required API keys as environment variables:
   ```bash
   export ANTHROPIC_API_KEY="your-claude-api-key"
   export WANDB_API_KEY="your-wandb-api-key"  # Optional, but recommended for logging
   ```

### Running the Pipeline

It is recommended to run the pipeline sequentially to ensure all required data is generated before evaluation. 

1. **Load and Prepare the Data:**
   ```bash
   python data_loader.py
   ```

2. **Generate Validation/Test QA Pairs:**
   *Note: This step requires Claude API access and is necessary for evaluation.*
   ```bash
   python qa_generator.py
   ```

3. **Execute the Full Phase-1 Pipeline End-to-End:**
   This step will train the Word2Vec model, evaluate the QA pairs using TinyLlama and Claude, and log the summary metrics.
   ```bash
   python run_phase1.py
   ```
   *(To run without saving metrics to Weights & Biases, run: `python run_phase1.py --no-wandb`)*

All configurations, including input data paths and model constraints, can be customized directly in the `config.yaml` file.