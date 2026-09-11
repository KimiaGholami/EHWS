"""Calibration and perplexity-eval data.

Calibration protocol matches the convention both background papers in
this proposal use for their baselines (ELSA appendix B.2: "we follow the
convention of Frantar & Alistarh (2023) [SparseGPT], sampling 128
calibration sequences from the C4 dataset with sequence length 2048"):

    128 sequences x 2048 tokens, drawn from C4's English "train" split,
    fixed seed for reproducibility.

Perplexity evaluation uses WikiText-2's test split (`wikitext-2-raw-v1`)
and a held-out slice of C4's validation split, matching ELSA's
Section B.1 evaluation protocol ("Perplexity is measured on the held-out
(validation) C4 ... and WikiText2").
"""

from __future__ import annotations

import random

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def get_c4_calibration(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """128 x 2048 token_id sequences from C4 train, SparseGPT/ELSA-style.

    Each sequence is a random contiguous `seqlen`-token crop from a
    random long-enough C4 document, sampled with a fixed seed so runs are
    exactly reproducible.
    """
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rng = random.Random(seed)

    samples = []
    it = iter(data)
    # Reservoir over a bounded prefix of the stream (C4 is enormous and
    # streamed; scanning a bounded number of documents keeps this fast
    # while still giving a fixed-seed, reproducible sample).
    pool_size = max(2000, n_samples * 20)
    pool = []
    for doc in it:
        text = doc["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] > seqlen:
            pool.append(ids)
        if len(pool) >= pool_size:
            break

    rng.shuffle(pool)
    for ids in pool:
        if len(samples) >= n_samples:
            break
        start = rng.randint(0, ids.shape[0] - seqlen - 1)
        samples.append(ids[start : start + seqlen])

    if len(samples) < n_samples:
        raise RuntimeError(f"Only found {len(samples)}/{n_samples} long-enough C4 documents")
    return torch.stack(samples, dim=0)


def get_wikitext2_test(tokenizer: PreTrainedTokenizerBase, seqlen: int = 2048) -> torch.Tensor:
    """WikiText-2 test split, concatenated and chunked into seqlen blocks."""
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.shape[0] // seqlen
    ids = ids[: n_chunks * seqlen]
    return ids.view(n_chunks, seqlen)


def get_c4_eval(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 256,
    seqlen: int = 2048,
    seed: int = 1,
) -> torch.Tensor:
    """A held-out C4 validation slice for perplexity, disjoint (different
    split, different seed) from calibration.
    """
    data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    rng = random.Random(seed)
    pool = []
    for doc in data:
        ids = tokenizer(doc["text"], return_tensors="pt").input_ids[0]
        if ids.shape[0] > seqlen:
            pool.append(ids)
        if len(pool) >= max(1000, n_samples * 10):
            break
    rng.shuffle(pool)
    samples = []
    for ids in pool:
        if len(samples) >= n_samples:
            break
        start = rng.randint(0, ids.shape[0] - seqlen - 1)
        samples.append(ids[start : start + seqlen])
    return torch.stack(samples, dim=0)
