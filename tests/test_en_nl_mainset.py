"""experiments/en-nl-translator/mainset.py on a made-up FLAN-nl plan: the eval rows are fixed and their segments stay
out of the replay pairs, and every filter of the replay counts what it drops. Run with --noconftest where the
conftest's imports (jax, flax) are not installed."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

import mainset  # noqa: E402

LONG_EN = "The committee discussed the proposal at length and decided to ask the minister for more information."
LONG_NL = ("De commissie besprak het voorstel uitgebreid en besloot de minister om meer informatie te vragen over de "
           "gevolgen voor de gemeenten en de provincies in het hele land.")


def make_plan(tmp_path, n=40):
    plan, en_dir, tr = tmp_path / "plan2", tmp_path / "en", tmp_path / "tr" / "w0" / "small"
    for d in (plan, en_dir, tr):
        d.mkdir(parents=True)
    texts = [f"{LONG_EN} Item {k}." for k in range(n)] + ["short one", "Loop here."]
    tokens = [60] * n + [3, 3]
    ids = list(range(len(texts)))
    pq.write_table(pa.table({"id": ids, "text": texts, "tokens": tokens}), plan / "segments-w0.parquet")
    examples = {"host": [0] * n, "position": list(range(n)), "source": ["flan/task_a"] * n,
                "status": ["translate"] * n,
                "instruction": [json.dumps([["S", k], ["L", "\n"]]) for k in range(n)],
                "answer": [json.dumps([["S", n]]) for _ in range(n)]}
    pq.write_table(pa.table(examples), plan / "examples.parquet")
    pq.write_table(pa.table({"host": [0] * n, "position": list(range(n)), "inputs": texts[:n], "targets": ["x"] * n}),
                   en_dir / "w0.parquet")
    nl = [f"{LONG_NL} Punt {k}." for k in range(n)] + ["kort", "Lus lus lus lus lus lus lus lus."]
    nl[1] = LONG_NL + "\n\nPunt 1."  # a paragraph break the English lacks
    nl[2] = "**" + LONG_NL + "**"  # bold the English lacks
    ended = [True] * len(texts)
    ended[3] = False
    pq.write_table(pa.table({"id": ids, "text_nl": nl, "ended": ended, "tokens_in": tokens,
                             "tokens_out": [61] * n + [2, 9]}), tr / "shard-00000.parquet")
    return plan, en_dir, tmp_path / "tr"


def test_eval_rows_and_replay(tmp_path, monkeypatch):
    plan, en_dir, tr = make_plan(tmp_path)
    monkeypatch.setattr(mainset, "EVAL_SETS", (("classifier", 4), ("yeb", 2)))
    mainset.main(["eval-rows", "--plan", str(plan), "--english", str(en_dir / "w0.parquet"),
                  "--out", str(tmp_path / "eval")])
    rows = pq.read_table(tmp_path / "eval" / "eval-rows.parquet").to_pylist()
    assert [r["set"] for r in rows] == ["classifier"] * 4 + ["yeb"] * 2
    assert all(r["inputs"].startswith(LONG_EN) for r in rows)
    again = tmp_path / "eval2"
    mainset.main(["eval-rows", "--plan", str(plan), "--english", str(en_dir / "w0.parquet"), "--out", str(again)])
    assert pq.read_table(again / "eval-rows.parquet").to_pylist() == rows  # the seed fixes them
    excluded = {i for r in rows for i in r["segments"]}
    mainset.main(["replay", "--plan", str(plan), "--translations", str(tr), "--eval-rows", str(tmp_path / "eval"),
                  "--out", str(tmp_path / "replay"), "--processes", "1"])
    report = json.loads((tmp_path / "replay" / "report.json").read_text())
    steps = report["steps"]
    assert steps["segments"] == 42
    assert steps["eval_rows"]["dropped"] == len(excluded)
    assert steps["not_ended"]["dropped"] == (0 if 3 in excluded else 1)
    pairs = pq.read_table(tmp_path / "replay" / "pairs.parquet").to_pylist()
    kept = {p["id"] for p in pairs}
    assert not kept & excluded and 41 not in kept and 40 not in kept  # the short ones are outside the band
    assert 1 not in kept and 2 not in kept  # their structure differs
    assert all(p["task"] == "flan/task_a" for p in pairs)
