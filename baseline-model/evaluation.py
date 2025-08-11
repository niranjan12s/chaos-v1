import torch
import torch.nn.functional as F
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

def compute_perplexity(model, dataloader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0][:, :-1].to(device)
            targets = batch[0][:, 1:].to(device)

            if hasattr(model, "dim_emb"):  # Chaos
                logits, _ = model(inputs)
            else:
                logits = model(inputs)

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl.item()

def generate_outputs(model, tokenizer, prompts, max_len=10, device='cpu'):
    model.eval()
    outputs = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            outputs.append([])
            continue
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        for _ in range(max_len):
            if hasattr(model, "dim_emb"):  # Chaos
                logits, _ = model(input_ids)
            else:
                logits = model(input_ids)
            next_token = torch.argmax(F.softmax(logits[:, -1], dim=-1), dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        full_output = input_ids.squeeze(0).tolist()
        outputs.append(full_output)
    return outputs

def compute_bleu(predictions, references, tokenizer):
    scores = []
    smoothie = SmoothingFunction().method4
    for pred_ids, ref in zip(predictions, references):
        if not pred_ids:
            scores.append(0)
            continue
        pred_words = tokenizer.decode(pred_ids).split()
        ref_words = ref.split()
        score = sentence_bleu([ref_words], pred_words, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0

def evaluate_model(model, dataloader, tokenizer, prompts, references, device='cpu'):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    ppl = compute_perplexity(model, dataloader, loss_fn, device)
    generated = generate_outputs(model, tokenizer, prompts, device=device)
    bleu = compute_bleu(generated, references, tokenizer)

    return {
        "perplexity": ppl,
        "bleu": bleu,
        "samples": generated,
    }


def encode(text, vocab):
    # The tokenize function expects a list of strings
    if isinstance(text, str):
        text = [text]
    tokens = tokenize(text, vocab)
    return tokens[0] # The tokenize function returns a list of lists

def decode(token_ids, rev_vocab):
    return ' '.join([rev_vocab[tid] for tid in token_ids])

class SimpleTokenizer:
    def __init__(self, vocab, rev_vocab):
        self.vocab = vocab
        self.rev_vocab = rev_vocab

    def encode(self, text):
        return encode(text, self.vocab)

    def decode(self, token_ids):
        return decode(token_ids, self.rev_vocab)

from torch.utils.data import DataLoader, TensorDataset

def create_eval_dataloader(tokenized_data, batch_size=4):
    data = []
    for line in tokenized_data:
        if len(line) < 2:
            continue
        tensor = torch.tensor(line, dtype=torch.long)
        data.append(tensor)

    padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return DataLoader(TensorDataset(padded), batch_size=batch_size)

lines = load_data("/content/training_set_chaos.txt")
vocab, inv_vocab = build_vocab(lines, vocab_size=1000)
tokenized_data = tokenize(lines, vocab=vocab)
eval_loader = create_eval_dataloader(tokenized_data)

def repetition_score(output_ids):
    scores = []
    for tokens in output_ids:
        if not tokens:
            scores.append({"uniq_unigram": 0, "uniq_bigram": 0})
            continue
        str_tokens = [str(t) for t in tokens]
        unigrams = len(set(str_tokens)) / len(str_tokens)
        if len(str_tokens) > 1:
            bigrams = len(set(zip(str_tokens, str_tokens[1:]))) / (len(str_tokens) - 1)
        else:
            bigrams = 0
        scores.append({"uniq_unigram": unigrams, "uniq_bigram": bigrams})
    return scores

from bert_score import score as bert_score

def compute_bertscore(predictions, references, lang="en"):
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    return F1.mean().item()

tokenizer = SimpleTokenizer(vocab, inv_vocab)
prompts = [
    "The dog chased the cat, but it",
    "She dropped the book because it",
    "Before the sun rose,",
    "After the lights went out,",
    "The keys were under the",
    "He hid the box behind the",
    "She looked at him with tears in her eyes and",
    "He stood there in silence,",
    "He claimed to love nature, but",
    "Although it was snowing,",
    "The glass fell from the table and",
    "He ran out of gas, so he",
    "The dragon flew over the mountains and",
    "The robot stared at the stars and",
    "He forgot her name, so",
    "The meeting ended and they",
    "The wind howled through the trees as",
    "No one expected her to"
]
references = [
    "escaped through the fence.",
    "was too heavy to carry.",
    "they packed their bags and left.",
    "the children screamed in fear.",
    "pillow on the left side of the bed.",
    "curtains in the living room.",
    "said nothing.",
    "unable to explain himself.",
    "he littered on the trail.",
    "she went out in sandals.",
    "shattered on the floor.",
    "walked to the nearest station.",
    "disappeared into the clouds.",
    "wondered what it meant to dream.",
    "he made up a story.",
    "walked away without a word.",
    "the storm rolled in.",
    "win the entire competition."
]
result = evaluate_model(
    model=model,
    dataloader=eval_loader,
    tokenizer=tokenizer,
    prompts=prompts,
    references=references,
    device="cpu" 
)

decoded_predictions = [tokenizer.decode(p) for p in result["samples"]]
bert_score_result = compute_bertscore(decoded_predictions, references)
repetition_score_result = repetition_score(result["samples"])
print(f"BERT SCORE: {bert_score_result}")
print(f"Repition Score: {repetition_score_result}")

print("Perplexity:", result["perplexity"])
print("BLEU:", result["bleu"])
for i, s in enumerate(result["samples"]):
    print(f"[{i}] {prompts[i]} {tokenizer.decode(s)}")

import numpy as np
uni=[x["uniq_unigram"] for x in repetition_score_result]
bi=[x["uniq_bigram"] for x in repetition_score_result]
print(f"Mean Uni-gram: {np.mean(uni)}")
print(f"Mean Bi-gram: {np.mean(bi)}")