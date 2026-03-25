# src/corrupt.py
import random
import spacy

# Run once: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def shuffle_within_nps(sentence: str) -> str:
    """Shuffles tokens only within each noun phrase. 
    Global word order is preserved — only local NP structure is broken."""
    doc = nlp(sentence)
    tokens = [t.text for t in doc]
    
    for chunk in doc.noun_chunks:
        start, end = chunk.start, chunk.end
        np_tokens = tokens[start:end]
        random.shuffle(np_tokens)
        tokens[start:end] = np_tokens
    
    return " ".join(tokens)

def shuffle_full_sentence(sentence: str) -> str:
    """Shuffles all tokens randomly. 
    Destroys both local NP structure and global syntactic order."""
    tokens = sentence.split()
    random.shuffle(tokens)
    return " ".join(tokens)

def apply_corruptions(input_path: str, output_dir: str, seed: int = 42):
    random.seed(seed)
    with open(input_path) as f:
        sentences = [line.strip() for line in f]
    
    np_shuffled   = [shuffle_within_nps(s)    for s in sentences]
    full_shuffled = [shuffle_full_sentence(s) for s in sentences]
    
    for name, data in [("original", sentences),
                       ("np_shuffled", np_shuffled),
                       ("full_shuffled", full_shuffled)]:
        with open(f"{output_dir}/{name}.txt", "w") as f:
            f.write("\n".join(data))
    
    print("Corruption complete.")

if __name__ == "__main__":
    apply_corruptions("data/raw/sentences.txt", "data/corrupted")