"""Build ELSA's tokenized C4 training set, for run_pipeline.py's --elsa-train-cache.

Run from inside a clone of ELSA's official code (https://github.com/log-postech/elsa),
with ELSA's own environment:

    cd elsa
    python /path/to/EHWS/semi-structured/scripts/build_elsa_train_cache.py --model fla-hub/hgrn-1.3B-100B

It calls ELSA's own `lib.data.get_dataset` with the settings ELSA uses for a
4096-step run (4096 steps x batch 1 x grad-accum 8 = 32,768 sequences of 2048
tokens, seed 0, slow tokenizer), so the tokens match ELSA's exactly. The
output directory is printed at the end, e.g.
`dataset/fla-hub_hgrn-1.3B-100B/train/c4_32768_2048_hub`. This is the same
directory `main.py --admm_save_inputs=True` writes, without the 5-hour ELSA
run. Tokenizing takes about 35 minutes.
"""

import argparse
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from lib.data import get_dataset  # noqa: E402  (ELSA's own code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fla-hub/hgrn-1.3B-100B")
    p.add_argument("--nsamples", type=int, default=32768)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="dataset")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    get_dataset(
        dataset_name="c4", tokenizer=tokenizer, nsamples=args.nsamples, seed=args.seed,
        seqlen=args.seqlen, data_type="train", cache_dir=args.cache_dir, save_to_cache=True,
    )
    out = os.path.join(args.cache_dir, args.model.replace("/", "_"), "train",
                       f"c4_{args.nsamples}_{args.seqlen}_hub")
    print(f"done: pass --elsa-train-cache {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
