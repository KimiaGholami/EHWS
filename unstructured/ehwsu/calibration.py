"""Calibration and perplexity-eval data.

Calibration follows the standard convention in this line of work
(SparseGPT, ELSA, etc.): 128 sequences of 2048 tokens each, drawn from
C4's English training split, with a fixed seed for reproducibility.

Perplexity is evaluated on WikiText-2's test split and a held-out slice
of C4's validation split, disjoint from whatever was used for
calibration.
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
    """n_samples x seqlen token_id sequences from C4's train split.

    Each sequence is a random contiguous seqlen-token crop from a random
    long-enough C4 document, sampled with a fixed seed so runs are
    reproducible.
    """
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rng = random.Random(seed)

    # C4 is streamed and effectively unbounded, so we build a candidate
    # pool from a bounded number of raw documents rather than scanning
    # until we've found enough that are long enough. Only a small
    # fraction of C4 documents clear the seqlen bar, so bounding by
    # "documents that qualified" instead of "documents scanned" makes the
    # scan time blow up as n_samples grows -- bounding the raw scan
    # directly keeps this predictable.
    max_raw_docs = max(50_000, n_samples * 100)
    pool = []
    n_scanned = 0
    for doc in data:
        n_scanned += 1
        ids = tokenizer(doc["text"], return_tensors="pt").input_ids[0]
        if ids.shape[0] > seqlen:
            pool.append(ids)
        if n_scanned >= max_raw_docs:
            break

    rng.shuffle(pool)
    samples = []
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
    """A held-out C4 validation slice for perplexity.

    Uses a different split and a different seed than calibration, so it's
    disjoint from whatever text calibration saw.
    """
    data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    rng = random.Random(seed)
    max_raw_docs = max(10_000, n_samples * 100)
    pool = []
    n_scanned = 0
    for doc in data:
        n_scanned += 1
        ids = tokenizer(doc["text"], return_tensors="pt").input_ids[0]
        if ids.shape[0] > seqlen:
            pool.append(ids)
        if n_scanned >= max_raw_docs:
            break
    rng.shuffle(pool)
    samples = []
    for ids in pool:
        if len(samples) >= n_samples:
            break
        start = rng.randint(0, ids.shape[0] - seqlen - 1)
        samples.append(ids[start : start + seqlen])
    return torch.stack(samples, dim=0)
