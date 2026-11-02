"""Data prep for the concept-roadmap experiment.

Reads TinyStories text files, trains a small byte-level BPE tokenizer, tokenizes
every story, and marks clause boundaries with a rule that only looks at the
current and previous token (so the model can apply it while generating).

Outputs, per split, in data/:
  {split}_tokens.bin  uint16 token ids, stories separated by <|endoftext|>
  {split}_flags.bin   uint8 per token: 0 = inside clause,
                      1 = clause ends here, 2 = sentence (or story) ends here
"""

import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
SENTENCE_END = set(".!?")
CONJUNCTIONS = {"and", "but", "so", "then", "because"}
QUOTES = "\"'”’"


def load_stories(path, truncated_tail):
    text = open(path, encoding="utf-8").read()
    chunks = [c.strip() for c in text.split(EOT)]
    # Files can start mid-story; a byte-range slice also ends mid-story.
    if chunks and chunks[0][:1].islower():
        chunks = chunks[1:]
    if truncated_tail:
        chunks = chunks[:-1]
    return [c for c in chunks if c]


def train_tokenizer(stories, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(stories, trainer)
    return tok


def boundary_tables(tok):
    """Per-token-id lookup tables used by the causal boundary rule."""
    n = tok.get_vocab_size()
    ends_sentence = np.zeros(n, bool)
    ends_comma = np.zeros(n, bool)
    is_conj = np.zeros(n, bool)
    for i in range(n):
        s = tok.decode([i], skip_special_tokens=False).rstrip()
        core = s.rstrip(QUOTES)
        if core[-1:] in SENTENCE_END:
            ends_sentence[i] = True
        elif core[-1:] in {";", ":"}:
            ends_comma[i] = True  # treated like ", <conj>" but fires immediately
        if s.endswith(","):
            ends_comma[i] = True
        if s.strip().lower() in CONJUNCTIONS:
            is_conj[i] = True
    return ends_sentence, ends_comma, is_conj


def mark_boundaries(ids, tables, eot_id):
    """Flags for one token stream. Each flag depends only on tokens <= t."""
    ends_sentence, ends_comma, is_conj = tables
    ids = np.asarray(ids)
    flags = np.zeros(len(ids), np.uint8)
    flags[ends_sentence[ids]] = 2
    flags[ids == eot_id] = 2
    # ", and" / ", but" ...: the clause boundary fires on the conjunction token.
    prev_comma = np.zeros(len(ids), bool)
    prev_comma[1:] = ends_comma[ids[:-1]]
    flags[(flags == 0) & prev_comma & is_conj[ids]] = 1
    return flags


def encode_split(tok, stories, tables):
    eot_id = tok.token_to_id(EOT)
    ids = []
    for enc in tok.encode_batch(stories):
        ids.extend(enc.ids)
        ids.append(eot_id)
    ids = np.array(ids, np.uint16)
    return ids, mark_boundaries(ids, tables, eot_id)


def clause_stats(flags):
    ends = np.flatnonzero(flags > 0)
    lengths = np.diff(np.concatenate([[-1], ends]))
    sent_ends = np.flatnonzero(flags == 2)
    sent_lengths = np.diff(np.concatenate([[-1], sent_ends]))
    pct = lambda a: {p: int(np.percentile(a, p)) for p in (10, 50, 90, 99)}
    return {
        "tokens": int(len(flags)),
        "clauses": int(len(ends)),
        "conjunction_boundaries": int((flags == 1).sum()),
        "clause_len_mean": round(float(lengths.mean()), 2),
        "clause_len_pct": pct(lengths),
        "sentence_len_mean": round(float(sent_lengths.mean()), 2),
        "sentence_len_pct": pct(sent_lengths),
    }


def show(tok, ids, flags, n_stories):
    eot_id = tok.token_to_id(EOT)
    out, seen = [], 0
    for i, f in zip(ids, flags):
        if i == eot_id:
            seen += 1
            out.append("\n\n")
            if seen == n_stories:
                break
            continue
        out.append(tok.decode([int(i)]))
        if f == 1:
            out.append(" |")
        elif f == 2:
            out.append(" ||")
    print("".join(out))


def check_causal(tok, stories, tables, n=200):
    """Flags on a prefix must match the flags of the full story at those positions."""
    eot_id = tok.token_to_id(EOT)
    for enc in tok.encode_batch(stories[:n]):
        full = mark_boundaries(enc.ids, tables, eot_id)
        for cut in (5, len(enc.ids) // 2, len(enc.ids) - 1):
            if cut > 0:
                assert (mark_boundaries(enc.ids[:cut], tables, eot_id) == full[:cut]).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train", default="train_slice.txt")
    ap.add_argument("--valid", default="valid.txt")
    ap.add_argument("--train-truncated", action="store_true",
                    help="train file is a byte slice, so drop its last story")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tokenizer-stories", type=int, default=200_000)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    d = args.data_dir
    train = load_stories(os.path.join(d, args.train), args.train_truncated)
    valid = load_stories(os.path.join(d, args.valid), False)
    print(f"stories: train {len(train):,}  valid {len(valid):,}")

    tok_path = os.path.join(d, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        tok = train_tokenizer(train[: args.tokenizer_stories], args.vocab_size)
        tok.save(tok_path)
    tables = boundary_tables(tok)
    check_causal(tok, valid, tables)

    meta = {"vocab_size": tok.get_vocab_size(), "eot_id": tok.token_to_id(EOT)}
    for split, stories in (("train", train), ("valid", valid)):
        ids, flags = encode_split(tok, stories, tables)
        ids.tofile(os.path.join(d, f"{split}_tokens.bin"))
        flags.tofile(os.path.join(d, f"{split}_flags.bin"))
        meta[split] = {"stories": len(stories), **clause_stats(flags)}
        print(split, json.dumps(meta[split]))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)

    if args.show:
        ids = np.fromfile(os.path.join(d, "valid_tokens.bin"), np.uint16)
        flags = np.fromfile(os.path.join(d, "valid_flags.bin"), np.uint8)
        show(tok, ids, flags, args.show)


if __name__ == "__main__":
    main()
