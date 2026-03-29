# src/data_prep.py
# Person A — Data preparation
# Pulls sentences from SST-2, filters by token length,
# and saves them to data/raw/sentences.txt.

import os
import random
import argparse
from datasets import load_dataset


SEED = 42
MIN_TOKENS = 10
MAX_TOKENS = 25
N_SENTENCES = 500   # default value; can be overridden with --n
OUTPUT_PATH = "data/raw/sentences.txt"


def load_sentences(n=N_SENTENCES, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, seed=SEED):
    """
    Load and filter sentences from SST-2 (HuggingFace Hub).

    Args:
        n: Number of sentences to collect.
        min_tokens: Minimum allowed token count.
        max_tokens: Maximum allowed token count.
        seed: Random seed for reproducibility.

    Returns:
        A list of filtered sentences.

    Notes:
        Filters by token count to ensure sentences have enough NP structure
        for the corruption step, but aren't so long they slow down inference.
    """
    random.seed(seed)

    print("Downloading SST-2 dataset from HuggingFace Hub...")
    dataset = load_dataset("stanfordnlp/sst2", split="train")

    sentences = []
    for item in dataset:
        sentence = item["sentence"].strip()
        tokens = sentence.split()

        if min_tokens <= len(tokens) <= max_tokens:
            sentences.append(sentence)

        if len(sentences) >= n:
            break

    if len(sentences) < n:
        print(
            f"[WARNING] Only found {len(sentences)} sentences matching filters "
            f"(min={min_tokens}, max={max_tokens} tokens). "
            f"Consider relaxing the token range."
        )
    else:
        print(f"Collected {len(sentences)} sentences.")

    return sentences


def save_sentences(sentences, path=OUTPUT_PATH):
    """
    Save sentences to a plain text file, one per line.
    Overwrites the file each time the script is run.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"Saved to {path}")


def validate_sentences(path=OUTPUT_PATH):
    """
    Print summary statistics and sample sentences.
    Run this after saving to confirm the data looks right
    before handing off to the corruption step.
    """
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if not sentences:
        print("[WARNING] No sentences found in file.")
        return

    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)

    print("\n--- Validation Report ---")
    print(f"Total sentences : {len(sentences)}")
    print(f"Avg token count : {avg:.1f}")
    print(f"Min token count : {min(lengths)}")
    print(f"Max token count : {max(lengths)}")
    print("\nSample sentences:")

    for s in sentences[:5]:
        print(f"  [{len(s.split())} tokens] {s}")

    print("-------------------------\n")

# number of sentences can now be chosen by running(1500 for example): python src/data_prep.py --n 1500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare filtered SST-2 sentences for the entropy experiment.")
    parser.add_argument("--n", type=int, default=N_SENTENCES, help="Number of sentences to collect.")
    parser.add_argument("--min_tokens", type=int, default=MIN_TOKENS, help="Minimum token count.")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Maximum token count.")
    args = parser.parse_args()

    sentences = load_sentences(
        n=args.n,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=SEED,
    )
    save_sentences(sentences)
    validate_sentences()
