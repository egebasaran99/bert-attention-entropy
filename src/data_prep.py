# src/data_prep.py
from datasets import load_dataset
import re

def load_sentences(n=500, min_tokens=10, max_tokens=25):
    dataset = load_dataset("sst2", split="train")
    sentences = []
    for item in dataset:
        sentence = item["sentence"].strip()
        tokens = sentence.split()
        if min_tokens <= len(tokens) <= max_tokens:
            sentences.append(sentence)
        if len(sentences) >= n:
            break
    return sentences

def save_sentences(sentences, path="data/raw/sentences.txt"):
    with open(path, "w") as f:
        for s in sentences:
            f.write(s + "\n")

if __name__ == "__main__":
    sents = load_sentences()
    save_sentences(sents)
    print(f"Saved {len(sents)} sentences.")