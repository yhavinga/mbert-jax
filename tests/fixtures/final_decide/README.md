# Fixtures of tests/test_final_decide.py: real results of G, X, F and R at step 300,000

Copied on 2026-11-10 from `gs://t5-dutch-english-usc2/mbert/` (byte for byte, except where noted), in the layout that
`experiments/final/decide.py --root` reads:

- `dumb/<name>--{dbrd,sicknl}--seed{42,43,44}.json`: `masking/dumb/` of `cxgptnl-300000` (G), `300000` (X),
  `fw2-300000` (F) and `fw2rh3-300000` (R); DBRD at 512 and SICK-NL, fine-tune LR 2e-5 (the DUMB protocol of the
  corpus runs, `decide_g.py` and `scripts/decide_r.py`).
- `dumb/<name>--dbrd1024--seed{42,43,44}.json`: `dbrd1024/dumb/` of G and X at LR 2e-5. X's files are
  `u15-300000--dbrd1024--seed*.json` in the bucket, renamed here to `300000--…` so that one name holds all of X's
  results (their `checkpoint` field still says `ckptu15-300000`).
- `dumb-extra/{squadnl,pos-lassysmall,pos-alpino}/<name>--<task>--seed*.json`: `dumb-extra/<task>/` of G and X.
- `blimp-nl/<name>.json`: `blimp-nl/` of G and X without the keys `pair_scores`, `paradigms` and `phenomena`.

Not here: CoNLL-2002. Its README allows research use in the context of the shared task only, and its numbers are not
published without Yeb; this repository is public. The tests put the same made-up CoNLL-2002 results under both arms.
