def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Strip whitespace and remove blank lines
    lines = [line.strip() for line in lines if line.strip()]
    return lines

from collections import Counter

def build_vocab(lines, vocab_size=1000):
    """
    Builds a vocabulary from tokenized lines.

    Parameters:
    - lines: list of strings or list of list-of-tokens (e.g., [["the", "cat"], ["sat", "there"]])
    - vocab_size: maximum number of words in the vocabulary

    Returns:
    - stoi: dict mapping token to index
    - itos: list mapping index to token
    """
    # Tokenize if not already tokenized
    tokenized = [line.split() if isinstance(line, str) else line for line in lines]

    # Flatten and count word frequencies
    counter = Counter(token for line in tokenized for token in line)

    # Keep top `vocab_size - 1` most frequent words, reserve index 0 for <UNK>
    most_common = counter.most_common(vocab_size - 1)
    vocab = [token for token, _ in most_common]

    # Add UNK at index 0
    itos = ["<UNK>"] + vocab
    stoi = {token: idx for idx, token in enumerate(itos)}

    return stoi, itos


def tokenize(lines, vocab):
    tokenized = []
    for line in lines:
        tokens = [vocab.get(word) for word in line.split()]
        tokens = [tok for tok in tokens if tok is not None]  # Filter out None values
        tokenized.append(tokens)
    return tokenized


import torch

def create_batches(tokenized, batch_size=8, pad_idx=0):
    import random
    random.shuffle(tokenized)
    batches = []
    for i in range(0, len(tokenized), batch_size):
        batch = tokenized[i:i+batch_size]
        max_len = max(len(seq) for seq in batch)
        padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in batch]
        batches.append(torch.tensor(padded))
    return batches
