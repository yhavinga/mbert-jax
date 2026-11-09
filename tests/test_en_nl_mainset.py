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


# ---------------------------------------------------------------- the final selection (Yeb, 9 Nov, design §7 point 7)

import collections  # noqa: E402
import types  # noqa: E402

import select_passages as sp  # noqa: E402


def cand(k, length="short", idiom=False, context=False, topic="A", host=None, source="raad", doc=None):
    return {"pid": f"{source}:{doc or k}:d:0", "source": source, "doc_id": str(doc or k), "length_class": length,
            "idiom_class": idiom, "context": context, "topic": topic, "host": host,
            "text": f"tekst nummer {k} " + " ".join(f"w{k}x{j}" for j in range(20))}


def test_quotas_hold_length_idiom_and_context_shares():
    cands = [cand(k, length="long" if k % 3 == 0 else "short", idiom=k % 2 == 0, context=k % 5 == 0,
                  source="handelingen") for k in range(400)]
    q = mainset.Quotas(40, 0.25, 0.20, None, None, sp.Chooser(15))
    chosen = q.draw(cands)
    assert len(chosen) == 40 and q.passes[1] == 40
    c = collections.Counter((p["length_class"], p["idiom_class"]) for p in chosen)
    assert c[("long", True)] == 5 and c[("long", False)] == 5 and c[("short", True)] == 15
    assert sum(p["context"] for p in chosen if p["length_class"] == "long") == 2
    assert sum(p["context"] for p in chosen if p["length_class"] == "short") == 6


def test_quotas_fill_a_short_idiom_half_in_a_later_pass():
    cands = [cand(k, idiom=k < 3) for k in range(100)]
    q = mainset.Quotas(20, 0.0, 0.0, None, None, sp.Chooser(3))
    chosen = q.draw(cands)
    assert len(chosen) == 20 and sum(p["idiom_class"] for p in chosen) == 3
    assert q.passes[1] == 13 and q.passes[2] == 7


def test_quotas_cap_topics_and_hosts_and_relax_only_the_topic():
    cands = [cand(k, topic="O" if k < 80 else "B", host="nl.wikipedia.org" if k % 2 else f"h{k}.nl",
                  source="fineweb") for k in range(100)]
    q = mainset.Quotas(20, 0.0, 0.0, 0.15, 0.10, sp.Chooser(3))
    chosen = q.draw(cands)
    hosts = collections.Counter(p["host"] for p in chosen)
    topics = collections.Counter(p["topic"] for p in chosen)
    assert hosts["nl.wikipedia.org"] <= 2  # 10% of 20, never relaxed
    assert len(chosen) == 20 and topics["B"] == 20 - topics["O"]
    assert q.passes[1] == 6 and q.passes[5] == 14  # 3 per topic, then the topic cap goes


def test_quotas_respect_the_document_cap_and_duplicates():
    cands = [cand(k, doc="same") for k in range(10)] + [cand(99)] * 3
    chosen = mainset.Quotas(10, 0.0, 0.0, None, None, sp.Chooser(3)).draw(cands)
    assert len(chosen) == 4  # 3 of one document, the duplicate once


def test_host_of_drops_www():
    assert mainset.host_of("https://www.Example.nl/pagina?x=1") == "example.nl"
    assert mainset.host_of("https://nl.wikipedia.org/wiki/X") == "nl.wikipedia.org"
    assert mainset.host_of("geen url") == ""


class WordTokenizer:
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    @classmethod
    def from_file(cls, path):
        return cls()

    def encode_batch(self, texts, add_special_tokens=False):
        return [self.Encoding(t.split()) for t in texts]


def write_pool(d, rows, turns=None, report=None):
    d.mkdir(parents=True)
    columns = sorted({k for r in rows for k in r})
    train = [r for r in rows if r["split"] == "train"]
    evals = [r for r in rows if r["split"] == "eval"]
    for name, part in (("passages.parquet", train), ("eval.parquet", evals)):
        if part:
            pq.write_table(pa.table({c: [r.get(c) for r in part] for c in columns}), d / name)
    if turns:
        pq.write_table(pa.table({c: [t[c] for t in turns] for c in turns[0]}), d / "context-turns.parquet")
    (d / "report.json").write_text(json.dumps(report or {}))


def words(k, n):
    return " ".join(f"woord{k}x{j}" for j in range(n))


def test_final_selection_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=WordTokenizer))
    rows, fw, scores = [], [], []

    def add(target, source, k, split="train", score=1.0, origin=None, text=None, **extra):
        pid = f"{source}:{k}:d:0" if source != "handelingen" else f"handelingen:{k}:t5:0"
        text = text or f"Eerste alinea {words(k, 30)}.\nTweede alinea {words(k + 1000, 30)}."
        target.append({"pid": pid, "source": source, "doc_id": str(k), "split": split, "text": text,
                       "tokens": len(text.split()), "idiom_class": k % 2 == 0, "context_example": False,
                       "context_turns": [], **extra})
        scores.append({"id": pid, "score": score, "skipped": None, "origin_score": origin, "topic": "ABCDE"[k % 5]})

    for k in range(60):
        add(rows, "handelingen", k, context_example=k < 10, context_turns=[3, 4] if k < 10 else [])
    for k in range(100, 140):
        add(rows, "raad", k, score=0.2 if k < 105 else 0.9)
    for source, base in (("dpc", 200), ("rechtspraak", 300)):
        for k in range(base, base + 10):
            add(rows, source, k)
    for k in range(400, 440):
        add(fw, "fineweb", k, origin=0.3 if k < 410 else 0.8)
    add(rows, "raad", 150, text="_Kop_ De tekst " + words(150, 30) + ".")
    for k in range(500, 505):
        add(rows, "raad", k, split="eval")
    turns = []
    for k in range(10):
        for j in (3, 4):
            turns.append({"id": f"handelingen:{k}:t{j}", "doc_id": str(k), "turn": j, "speaker": "De voorzitter",
                          "party": None, "text": f"Beurt {j}.\nNog een alinea {words(k + 50 * j, 5)}.", "tokens": 9,
                          "used_by": 1})
            # turn 4 of debate 0 is rejected on its source: its context ends before it, so nothing is left
            scores.append({"id": f"handelingen:{k}:t{j}", "score": 0.1 if (k, j) == (0, 4) else 1.0,
                           "skipped": None, "origin_score": None, "topic": "A"})
    write_pool(tmp_path / "pool", rows, turns, {"args": {"max_per_doc": "handelingen=15,raad=5"}})
    write_pool(tmp_path / "pool-fineweb", fw, None, {"args": {"max_per_doc": ""}})
    (tmp_path / "scores" / "w0").mkdir(parents=True)
    pq.write_table(pa.table({c: [s[c] for s in scores] for c in scores[0]}),
                   tmp_path / "scores" / "w0" / "shard-00000.parquet")
    pq.write_table(pa.table({"doc_id": [str(k) for k in range(400, 440)],
                             "url": [f"https://www.site{k % 20}.nl/p" for k in range(400, 440)],
                             "host": [f"site{k % 20}.nl" for k in range(400, 440)]}), tmp_path / "hosts.parquet")
    out = tmp_path / "final"
    mainset.main(["final", "--select", str(tmp_path / "pool"), str(tmp_path / "pool-fineweb"), "--scores",
                  str(tmp_path / "scores"), "--hosts", str(tmp_path / "hosts.parquet"), "--tokenizer", "x",
                  "--n", "100", "--n-eval", "5", "--max-tokens", "768", "--out", str(out)])
    chosen = pq.read_table(out / "passages.parquet").to_pylist()
    report = json.loads((out / "report.json").read_text())
    by = collections.Counter((p["source"], p["split"]) for p in chosen)
    # FineWeb-2 wants 25, but its 30 candidates come from 20 hosts and a host may give 1 (0.5% of 25, at least 1)
    assert by[("handelingen", "train")] == 40 and by[("fineweb", "train")] == 20
    assert report["draws"]["fineweb/train"]["passes"]["5"] == 5  # 3 per topic first, then the topic cap goes
    assert all(p["source_score"] >= 0.5 for p in chosen)
    assert all(p["origin_score"] >= 0.5 for p in chosen if p["source"] == "fineweb")
    assert report["eligibility"]["fineweb/train"]["origin_below"] == 10
    assert report["fineweb_origin"] == {"passages": 40, "below_0.1": 0, "below_0.5": 10, "below_0.9": 40}
    assert report["eligibility"]["raad/train"]["source_below"] == 5
    # an empty line between paragraphs, the marked heading on a line of its own
    assert all("\n\n" in p["text"] and "\n\n" not in p["text_pool"] for p in chosen if p["text_pool"].count("\n"))
    kop = [p for p in chosen if p["pid"] == "raad:150:d:0"]
    assert kop and kop[0]["text"].startswith("_Kop_\n\nDe tekst") and kop[0]["headings_repaired"] == 1
    # the context: debate 0 keeps nothing (its nearest turn failed), the others both turns
    ctx = {p["pid"]: p["context_ids"] for p in chosen if p["source"] == "handelingen"}
    assert "handelingen:0:t5:0" not in ctx or ctx["handelingen:0:t5:0"] == []
    assert all(v == [f"handelingen:{p.split(':')[1]}:t3", f"handelingen:{p.split(':')[1]}:t4"]
               for p, v in ctx.items() if v)
    turns_out = pq.read_table(out / "context-turns.parquet").to_pylist()
    assert {t["id"] for t in turns_out} == {i for v in ctx.values() for i in v}
    assert all("\n\n" in t["text"] for t in turns_out)
    # FineWeb-2: no host above its cap (0.5% of 25, so 1)
    hosts = collections.Counter(p["host"] for p in chosen if p["source"] == "fineweb")
    assert max(hosts.values()) == 1
    segments = [r for k in range(4) for r in pq.read_table(out / f"segments-w{k}.parquet").to_pylist()]
    assert len({r["id"] for r in segments}) == len(segments) == len(chosen) + len(turns_out)
    assert (out / "selection.md").read_text().startswith("# Hoofdset")
