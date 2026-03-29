# src/data_prep.py
# Person A — Data preparation
# Pulls sentences from a selected source dataset, filters by token length,
# and saves them to data/raw/sentences.txt.
#
# Supported datasets:
#   - sst2
#   - babylm
#
# Example usage:
#   python src/data_prep.py --dataset sst2 --n 1000
#   python src/data_prep.py --dataset babylm --n 1000 --babylm_dir data/babylm_clean_10M
#   python src/data_prep.py --dataset babylm --n 2000 --babylm_dir data/babylm_clean_10M --balanced_babylm

import os
import re
import random
import argparse
from pathlib import Path

import spacy
from datasets import load_dataset


SEED = 42
DEFAULT_DATASET = "sst2"
MIN_TOKENS = 10
MAX_TOKENS = 25
N_SENTENCES = 500
OUTPUT_PATH = "data/raw/sentences.txt"


def clean_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_sentence(sentence: str, min_tokens: int, max_tokens: int) -> bool:
    tokens = sentence.split()

    if len(tokens) < min_tokens or len(tokens) > max_tokens:
        return False

    if not re.search(r"[A-Za-z]", sentence):
        return False

    return True


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


# ---------------------------------------------------------------------------
# SST-2
# ---------------------------------------------------------------------------

def load_sst2_sentences(
    n: int,
    min_tokens: int,
    max_tokens: int,
    seed: int = SEED,
):
    """
    Load and filter sentences from SST-2.
    SST-2 already provides one sentence per example.
    """
    random.seed(seed)

    print("Downloading SST-2 dataset from HuggingFace Hub...")
    dataset = load_dataset("stanfordnlp/sst2", split="train")

    sentences = []
    for item in dataset:
        sentence = clean_sentence(item["sentence"])

        if is_valid_sentence(sentence, min_tokens, max_tokens):
            sentences.append(sentence)

        if len(sentences) >= n:
            break

    if len(sentences) < n:
        print(
            f"[WARNING] Only found {len(sentences)} SST-2 sentences matching filters "
            f"(min={min_tokens}, max={max_tokens} tokens)."
        )
    else:
        print(f"Collected {len(sentences)} SST-2 sentences.")

    return sentences


# ---------------------------------------------------------------------------
# BabyLM helpers
# ---------------------------------------------------------------------------

def get_babylm_txt_files(babylm_dir: str):
    base_path = Path(babylm_dir)

    if not base_path.exists():
        raise FileNotFoundError(
            f"BabyLM directory not found: {babylm_dir}\n"
            f"Pass a valid path with --babylm_dir"
        )

    txt_files = sorted(base_path.rglob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under: {babylm_dir}")

    return txt_files


def collect_sentences_from_text(
    text: str,
    nlp,
    min_tokens: int,
    max_tokens: int,
    limit: int | None = None,
):
    """
    Segment text into sentences and filter by token length.
    Uses chunking to avoid huge-doc issues on large BabyLM files.
    """
    sentences = []
    chunk_size = 200_000

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        doc = nlp(chunk)

        for sent in doc.sents:
            sentence = clean_sentence(sent.text)

            if is_valid_sentence(sentence, min_tokens, max_tokens):
                sentences.append(sentence)

                if limit is not None and len(sentences) >= limit:
                    return sentences

    return sentences


def load_babylm_sentences(
    babylm_dir: str,
    n: int,
    min_tokens: int,
    max_tokens: int,
    seed: int = SEED,
    balanced: bool = False,
):
    """
    Load and filter sentences from a local BabyLM text directory.

    If balanced=False:
        collects sentences across files in sorted order until n is reached.

    If balanced=True:
        tries to collect roughly equal numbers from each file.
        This is often better for experiments because no single source dominates.
    """
    random.seed(seed)

    txt_files = get_babylm_txt_files(babylm_dir)
    print(f"Loading BabyLM text files from: {babylm_dir}")
    print(f"Found {len(txt_files)} text files.")

    # Lightweight sentence segmenter
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    if not balanced:
        sentences = []

        for file_path in txt_files:
            print(f"  Reading: {file_path}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            file_sentences = collect_sentences_from_text(
                text=text,
                nlp=nlp,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                limit=None,
            )

            sentences.extend(file_sentences)

            if len(sentences) >= n:
                sentences = sentences[:n]
                print(f"Collected {len(sentences)} BabyLM sentences.")
                return sentences

        print(
            f"[WARNING] Only found {len(sentences)} BabyLM sentences matching filters "
            f"(min={min_tokens}, max={max_tokens} tokens)."
        )
        return sentences

    # Balanced sampling across files
    per_file_target = max(1, n // len(txt_files))
    remainder = n % len(txt_files)

    all_sentences = []

    for i, file_path in enumerate(txt_files):
        file_target = per_file_target + (1 if i < remainder else 0)

        print(f"  Reading: {file_path} (target={file_target})")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        file_sentences = collect_sentences_from_text(
            text=text,
            nlp=nlp,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            limit=None,
        )

        if len(file_sentences) > file_target:
            random.shuffle(file_sentences)
            file_sentences = file_sentences[:file_target]

        all_sentences.extend(file_sentences)

    # Final shuffle so sources are mixed
    random.shuffle(all_sentences)

    if len(all_sentences) < n:
        print(
            f"[WARNING] Only found {len(all_sentences)} BabyLM sentences matching filters "
            f"(min={min_tokens}, max={max_tokens} tokens) under balanced sampling."
        )
        return all_sentences

    all_sentences = all_sentences[:n]
    print(f"Collected {len(all_sentences)} BabyLM sentences (balanced across files).")
    return all_sentences


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_sentences(
    dataset_name: str,
    n: int = N_SENTENCES,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    seed: int = SEED,
    babylm_dir: str | None = None,
    balanced_babylm: bool = False,
):
    dataset_name = dataset_name.lower()

    if dataset_name == "sst2":
        return load_sst2_sentences(
            n=n,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            seed=seed,
        )

    if dataset_name == "babylm":
        if not babylm_dir:
            raise ValueError(
                "BabyLM selected but no directory was provided. "
                "Use --babylm_dir PATH"
            )

        return load_babylm_sentences(
            babylm_dir=babylm_dir,
            n=n,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            seed=seed,
            balanced=balanced_babylm,
        )

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        f"Choose from: sst2, babylm"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare filtered sentences for the entropy experiment."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=["sst2", "babylm"],
        help="Source dataset to use."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_SENTENCES,
        help="Number of sentences to collect."
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=MIN_TOKENS,
        help="Minimum token count."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=MAX_TOKENS,
        help="Maximum token count."
    )
    parser.add_argument(
        "--babylm_dir",
        type=str,
        default=None,
        help="Path to local BabyLM text directory (required if --dataset babylm)."
    )
    parser.add_argument(
        "--balanced_babylm",
        action="store_true",
        help="For BabyLM, sample approximately equally from each source file."
    )

    args = parser.parse_args()

    sentences = load_sentences(
        dataset_name=args.dataset,
        n=args.n,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=SEED,
        babylm_dir=args.babylm_dir,
        balanced_babylm=args.balanced_babylm,
    )

    save_sentences(sentences)
    validate_sentences()
