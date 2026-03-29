# src/inference.py
# Person C — BERT inference
# Runs all three sentence conditions through BERT-base-uncased,
# extracts per-layer attention entropy, and saves results to JSON.

import json
import os
import sys
import argparse

import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

# entropy.py must be in the same directory (src/)
sys.path.append(os.path.dirname(__file__))
from entropy import compute_entropy_per_layer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
OUTPUT_PATH = "results/entropy_results.json"

CONDITION_PATHS = {
    "original": "data/corrupted/original.txt",
    "np_shuffled": "data/corrupted/np_shuffled.txt",
    "full_shuffled": "data/corrupted/full_shuffled.txt",
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name=MODEL_NAME):
    """
    Loads BERT-base-uncased with attention outputs enabled.
    Automatically uses GPU if available, otherwise falls back to CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    model.to(device)

    return tokenizer, model, device


def run_inference(
    condition_paths=CONDITION_PATHS,
    output_path=OUTPUT_PATH,
    model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
):
    """
    For each condition:
      1. Loads sentences from disk.
      2. Runs each sentence through BERT.
      3. Computes per-layer attention entropy.
      4. Saves results incrementally to JSON after each condition.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(model_name)
    results = {}

    for condition_name, path in condition_paths.items():
        print(f"\nProcessing condition: {condition_name}")

        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {path}. Skipping.")
            continue

        with open(path, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f"  Loaded {len(sentences)} sentences.")
        all_layer_entropies = []

        with torch.no_grad():
            for sentence in tqdm(sentences, desc=f"  {condition_name}"):
                if not sentence:
                    all_layer_entropies.append([None] * 12)
                    continue

                try:
                    inputs = tokenizer(
                        sentence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=False,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model(**inputs)
                    layer_entropies = compute_entropy_per_layer(outputs.attentions)
                    all_layer_entropies.append(layer_entropies)

                except Exception as e:
                    print(f"\n  [ERROR] Failed on sentence: '{sentence[:60]}...' — {e}")
                    all_layer_entropies.append([None] * 12)

        results[condition_name] = all_layer_entropies

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"  Saved progress to {output_path}")

    print(f"\nAll conditions complete. Final results at: {output_path}")
    print_summary(results)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    """
    Prints a quick sanity check after inference:
    how many sentences succeeded vs failed per condition.
    """
    print("\n--- Inference Summary ---")
    for condition, entropies in results.items():
        total = len(entropies)
        failed = sum(1 for e in entropies if e[0] is None)
        success = total - failed
        print(f"  {condition:<15}: {success}/{total} succeeded, {failed} failed")
    print("-------------------------\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT attention entropy inference.")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to save JSON results.")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Maximum tokenized sequence length.")
    args = parser.parse_args()

    run_inference(
        output_path=args.output_path,
        max_length=args.max_length,
    )
