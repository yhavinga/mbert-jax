"""G4-8.15 (protocol-g4.md §19): g4mixsample.py (the 60 novel and 60 BT sources with their checks, the pick of the 40
pairs, the sample file and its key, the blind package) on small synthetic inputs, and run_g4mixsample.sh's use of the
evaluation's translate_class (no model, no TPU)."""

import argparse
import datetime
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4arm  # noqa: E402
import g4mixsample as mx  # noqa: E402
import g4rl  # noqa: E402
import pairjudge  # noqa: E402
from test_en_nl_g4rl import FakeCounter7  # noqa: E402

WORDS = ("the", "house", "river", "green", "walks", "slowly", "and", "city", "old", "light", "water", "sees", "a", "dog",
         "runs", "far", "away", "from", "home", "today", "bright", "stone", "bridge", "under", "over", "night")
SOURCES = ("dpc", "fineweb", "handelingen", "raad", "rechtspraak")


def sentence(rng, n=24):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + f" w{rng.randrange(10 ** 9)}"


# ---------------------------------------------------------------------------------------------------- data

def novels_inputs(rng):
    meta, b1 = [], []
    for k in range(mx.NOVEL_PAIRS):
        book = str(900 + k % mx.NOVEL_BOOKS)
        pid = str(10_000 + k)
        meta.append({"pid": pid, "en_book_id": book})
        b1.append({"id": pid, "group": "en_orig", "en_book_id": book, "en": "  " + sentence(rng) + " "})
    b1.append({"id": "99", "group": "nl_orig", "en_book_id": "1", "en": sentence(rng)})
    return meta, b1


def test_novel_rows_have_cands7s_ids_and_stop_on_a_missing_or_foreign_row():
    rng = random.Random(1)
    meta, b1 = novels_inputs(rng)
    rows = mx.novel_rows(meta, b1)
    assert len(rows) == 500 and {r["stratum"] for r in rows} == {str(900 + k) for k in range(10)}
    assert rows[0]["id"] == "novels:10000" and rows[0]["doc"] == "novels:900" and rows[0]["en"] == rows[0]["en"].strip()
    with pytest.raises(SystemExit, match="rows in b1-selected"):
        mx.novel_rows(meta, b1[1:])
    bad = [dict(r) for r in b1]
    bad[3]["en_book_id"] = "123"
    with pytest.raises(SystemExit, match="not en_orig of book"):
        mx.novel_rows(meta, bad)
    with pytest.raises(SystemExit, match="not 500 of 10"):
        mx.novel_rows(meta[:-1], b1)


def fb_pairs(rng):
    return [{"test": "bt", "id": f"{s}:doc{k}:d:0", "stratum": s, "en": sentence(rng), "first": "x", "second": "y"}
            for s in SOURCES for k in range(100)]


def write_inputs(tmp, rng, meta, b1, blocked_rows):
    paths = {}
    g4rl.write_parquet(meta, tmp / "meta.parquet")
    g4rl.write_parquet(b1, tmp / "b1.parquet")
    for name, rows in blocked_rows.items():
        g4rl.write_parquet(rows, tmp / f"{name}.parquet")
        paths[name] = tmp / f"{name}.parquet"
    g4rl.write_parquet([{"pid": "p", "source": "dpc", "split": "train", "label": 1, "text": "t"}], tmp / "c2.parquet")
    return paths


def data_args(tmp, paths):
    return argparse.Namespace(
        novels_meta=str(tmp / "meta.parquet"), b1=str(tmp / "b1.parquet"), g47_pool=str(paths["pool"]),
        g48_fresh=str(paths["fresh"]), g48_plan=str(paths["plan"]), g48_targets=str(paths["targets"]),
        c2_data=str(tmp / "c2.parquet"), bt=str(tmp / "bt"), tokenizer="unused", out=str(tmp / "out"))


@pytest.fixture
def fake_bt(monkeypatch):
    rng = random.Random(7)
    pairs = fb_pairs(rng)
    seen = {}

    def fb_bt(c2_rows, bt_rows, pool_rows, plan_rows):
        seen["pool"], seen["plan"] = len(pool_rows), len(plan_rows)
        return pairs, {"pairs": len(pairs)}

    monkeypatch.setattr(pairjudge, "fb_bt", fb_bt)
    monkeypatch.setattr(g4arm, "revisions3", lambda d: {})
    monkeypatch.setattr(g4arm, "read_bt", lambda *a, **k: [])
    return pairs, seen


def blocked_tables(rng, extra=None):
    rows = [{"id": f"novels:{k}", "doc": "novels:5", "en": sentence(rng)} for k in range(20)]
    out = {"pool": rows, "fresh": [dict(r, id=f"x{k}") for k, r in enumerate(rows)], "plan": rows[:5],
           "targets": rows[:3]}
    if extra:
        name, row = extra
        out[name] = out[name] + [row]
    return out


def test_data_draws_6_per_book_and_12_per_source_checked_and_filtered(tmp_path, fake_bt, capsys):
    pairs, seen = fake_bt
    rng = random.Random(3)
    meta, b1 = novels_inputs(rng)
    # a novel passage that shares a 13-gram with G4-8's targets is filtered (counted), not drawn
    gram = " ".join(WORDS[:14])
    b1[0]["en"] = gram + " and more words here today again and again"
    tables = blocked_tables(rng)
    tables["targets"] = tables["targets"] + [{"id": "z", "doc": "z", "en": "before " + gram}]
    paths = write_inputs(tmp_path, rng, meta, b1, tables)
    mx.data(data_args(tmp_path, paths), counter=FakeCounter7())
    out = tmp_path / "out"
    info = json.loads((out / "data.json").read_text())
    assert info["novels"]["drawn"] == 60 and info["bt"]["drawn"] == 60
    assert info["novels"]["filter"]["ngram"] == 1 and "novels:10000" not in info["novels"]["ids"]
    assert info["novels"]["overlap"] == {"id": 0, "doc": 0, "en": 0}
    assert seen == {"pool": 20, "plan": 5}
    ev = pq.read_table(out / "evalsets.parquet").to_pylist()
    seg = pq.read_table(out / "segments.parquet").to_pylist()
    assert [r["id"] for r in ev] == [r["id"] for r in seg] and len(ev) == 120
    per = {}
    for r in ev:
        per.setdefault((r["set"], json.loads(r["meta"])["stratum"]), 0)
        per[(r["set"], json.loads(r["meta"])["stratum"])] += 1
    assert {k: v for k, v in per.items() if k[0] == "novels"} == {("novels", str(900 + k)): 6 for k in range(10)}
    assert {k: v for k, v in per.items() if k[0] == "bt"} == {("bt", s): 12 for s in SOURCES}
    assert all(r["length_class"] == g4arm.class_of(r["tokens"]) and r["reference"] is None for r in ev)
    assert all(s["text"] == r["source"] and s["tokens"] == r["tokens"] for s, r in zip(seg, ev))
    assert (out / "classes.txt").read_text().split() == ["tiny"]
    printed = capsys.readouterr().out
    assert printed.startswith("MIXDATA ") and not any(r["source"] in printed for r in ev)
    # the same inputs give the same draw
    again = tmp_path / "again"
    a = data_args(tmp_path, paths)
    a.out = str(again)
    mx.data(a, counter=FakeCounter7())
    assert json.loads((again / "data.json").read_text())["novels"]["ids"] == info["novels"]["ids"]


@pytest.mark.parametrize("name, kind", [("pool", "id"), ("fresh", "doc"), ("targets", "en"), ("plan", "id")])
def test_data_stops_on_a_source_of_g47s_pool_or_g48s_files(tmp_path, fake_bt, name, kind):
    rng = random.Random(4)
    meta, b1 = novels_inputs(rng)
    hit = {"id": "z", "doc": "z", "en": "unrelated text"}
    if kind == "id":
        hit["id"] = "novels:10007"
    elif kind == "doc":
        hit["doc"] = "novels:903"
    else:
        hit["en"] = b1[11]["en"].strip()
    paths = write_inputs(tmp_path, rng, meta, b1, blocked_tables(rng, (name, hit)))
    with pytest.raises(SystemExit, match="novels: sources in G4-7's pool or G4-8's fresh, plan or targets"):
        mx.data(data_args(tmp_path, paths), counter=FakeCounter7())
    assert not (tmp_path / "out" / "data.json").exists()


def test_data_stops_on_a_bt_source_of_g48s_fresh(tmp_path, fake_bt):
    pairs, _ = fake_bt
    rng = random.Random(5)
    meta, b1 = novels_inputs(rng)
    hit = {"id": "z", "doc": "dpc:doc3", "en": "unrelated"}
    paths = write_inputs(tmp_path, rng, meta, b1, blocked_tables(rng, ("fresh", hit)))
    with pytest.raises(SystemExit, match="bt: sources in"):
        mx.data(data_args(tmp_path, paths), counter=FakeCounter7())


def test_draw_takes_per_stratum_a_seeded_shuffle_of_the_sorted_ids():
    rows = [{"id": f"s{k}", "stratum": "a" if k % 2 else "b"} for k in range(30)]
    out, eligible = mx.draw(rows, 4, lambda s: f"seed/{s}")
    assert eligible == {"a": 15, "b": 15} and len(out) == 8
    want_a = sorted(r["id"] for r in rows if r["stratum"] == "a")
    random.Random("seed/a").shuffle(want_a)
    assert [r["id"] for r in out[:4]] == want_a[:4]
    with pytest.raises(SystemExit, match="eligible sources, 16 wanted"):
        mx.draw(rows, 16, lambda s: s)


# ---------------------------------------------------------------------------------------------------- pick

def eval_fixture(tmp, rng):
    """The evaluation's evalsets (with blind passages) and both models' pairs, the split, the keys"""
    ev, assign = [], {}
    for s, n in (("flan", 40), ("wmt24pp", 40), ("flores", 30), ("ntrex", 30), ("europarl", 5)):
        for k in range(n):
            i = f"{s}:{k:03d}" if s == "flan" else f"{s}-{k:04d}"
            meta = json.dumps({"is_bad_source": k == 4}) if s == "wmt24pp" else None
            ev.append({"id": i, "set": s, "meta": meta, "source": sentence(rng)})
            if s != "europarl":
                assign[i] = "search" if k % 2 == 0 else "decision"
    for k in range(30):
        bs = ("flan", "wmt24pp", "idiom")[k % 3]
        ev.append({"id": f"blind-{k + 1:03d}", "set": "blind", "source": sentence(rng),
                   "meta": json.dumps({"blind_set": bs, "item": f"flan:{k:03d}" if bs == "flan" else f"gv:{k}",
                                       "task": "t"})})
    digest = hashlib.sha256(json.dumps(assign, sort_keys=True).encode()).hexdigest()
    (tmp / "split.json").write_text(json.dumps({"assign": assign, "digest": digest}))
    g4rl.write_parquet([{k: r[k] for k in ("id", "set", "meta")} for r in ev], tmp / "evalsets.parquet")
    base, own = [], []
    for k, r in enumerate(ev):
        hyp = sentence(rng)
        same = k % 4 == 3
        row = {"id": r["id"], "set": r["set"], "source": r["source"], "ended": True, "decode_loop": False,
               "empty": False}
        base.append(dict(row, hypothesis=hyp))
        own.append(dict(row, hypothesis=("  " + hyp.replace(" ", "  ")) if same else sentence(rng)))
    for m, rows in (("base", base), ("own", own)):
        (tmp / m).mkdir()
        g4rl.write_parquet(rows[:70], tmp / m / "pairs-w0.parquet")
        g4rl.write_parquet(rows[70:], tmp / m / "pairs-w1.parquet")
    keys = tmp / "keys"
    keys.mkdir()
    (keys / "replica.jsonl").write_text("".join(json.dumps({"item": i}) + "\n" for i in ("flan:000", "wmt24pp-0002")))
    (keys / "sample2.json").write_text(json.dumps([{"n": 1, "blind_id": "blind-003"}, {"n": 2, "blind_id": "blind-006"}]))
    return ev, assign, digest


def mix_fixture(tmp, rng):
    data = tmp / "data"
    data.mkdir()
    ev, base, own = [], [], []
    for part, strata, per in (("novels", [str(900 + k) for k in range(10)], 6), ("bt", SOURCES, 12)):
        for s in strata:
            for k in range(per):
                i = f"{part}:{s}:{k}"
                src = sentence(rng)
                ev.append({"id": i, "set": part, "meta": json.dumps({"stratum": s, "doc": s})})
                hyp = sentence(rng)
                row = {"id": i, "set": part, "source": src, "ended": k != 2, "decode_loop": False, "empty": False}
                base.append(dict(row, hypothesis=hyp))
                same = k % 3 == 0 or (part == "novels" and s == "900")
                own.append(dict(row, hypothesis=hyp if same else sentence(rng)))
    g4rl.write_parquet(ev, data / "evalsets.parquet")
    g4rl.write_parquet(base, tmp / "mix-base.parquet")
    g4rl.write_parquet(own, tmp / "mix-own.parquet")
    return {r["id"]: r for r in base}, {r["id"]: r for r in own}


def pick_args(tmp):
    return argparse.Namespace(
        data=str(tmp / "data"), evalsets=str(tmp / "evalsets.parquet"), split=str(tmp / "split.json"),
        used_blind=str(tmp / "keys" / "sample2.json"), out=str(tmp / "out"),
        mix=[f"base={tmp / 'mix-base.parquet'}", f"G4-8-s140={tmp / 'mix-own.parquet'}"],
        eval=[f"base={tmp / 'base' / 'pairs-w*.parquet'}", f"G4-8-s140={tmp / 'own' / 'pairs-w*.parquet'}"],
        exclude_items=[str(tmp / "keys" / "replica.jsonl")])


def parse_sample(text):
    blocks = re.split(r"^## (\d+)\n", text, flags=re.M)
    out = {}
    for n, body in zip(blocks[1::2], blocks[2::2]):
        m = re.fullmatch(r"\n\*\*Bron:\*\*\n\n(.*)\n\n\*\*A:\*\*\n\n(.*)\n\n\*\*B:\*\*\n\n(.*?)\n*", body, flags=re.S)
        assert m, body
        out[int(n)] = m.groups()
    return blocks[0], out


def test_pick_makes_40_blind_pairs_from_all_sources_and_the_package(tmp_path, monkeypatch, capsys):
    rng = random.Random(11)
    ev, assign, digest = eval_fixture(tmp_path, rng)
    monkeypatch.setattr(g4arm, "SPLIT_DIGEST6C", digest)
    mbase, mown = mix_fixture(tmp_path, rng)
    mx.pick(pick_args(tmp_path), today=datetime.date(2026, 11, 24))
    out = tmp_path / "out"
    key = json.loads((out / "key.json").read_text())
    head, pairs = parse_sample((out / mx.SAMPLE_FILE).read_text(encoding="utf-8"))
    assert "24 nov 2026" in head and '"A"' in head and '"="' in head and "fout A" in head
    assert sorted(pairs) == list(range(1, 41)) and len(key["pairs"]) == 40
    assert key["per_set"] == mx.PICK
    # no set name in the file for Yeb
    text = (out / mx.SAMPLE_FILE).read_text(encoding="utf-8")
    for name in ("novels", "bt:", "flan", "wmt24pp", "flores", "ntrex", "idiom", "romans", "G4-8-s140 is", "blind-"):
        assert name not in text.split("\n## 1\n", 1)[1]
    evr = {r["id"]: r for r in ev}
    blind_flan = {json.loads(r["meta"])["item"] for r in ev if r["set"] == "blind"}
    assert "flan:006" in blind_flan
    base = {**{r["id"]: r for r in pq.read_table(tmp_path / "base" / "pairs-w0.parquet").to_pylist()},
            **{r["id"]: r for r in pq.read_table(tmp_path / "base" / "pairs-w1.parquet").to_pylist()}, **mbase}
    own = {**{r["id"]: r for r in pq.read_table(tmp_path / "own" / "pairs-w0.parquet").to_pylist()},
           **{r["id"]: r for r in pq.read_table(tmp_path / "own" / "pairs-w1.parquet").to_pylist()}, **mown}
    for k in key["pairs"]:
        src, a, b = pairs[k["n"]]
        i = k["item"]
        assert src == base[i]["source"].strip()
        g48, g40 = (a, b) if k["g48"] == "A" else (b, a)
        assert g48 == own[i]["hypothesis"].strip() and g40 == base[i]["hypothesis"].strip()
        assert mx.norm(g48) != mx.norm(g40)
        assert (k["A"], k["B"]) == (("G4-8-s140", "G4-0") if k["g48"] == "A" else ("G4-0", "G4-8-s140"))
        assert k["g48"] == ("A" if random.Random(f"{mx.SEED}/ab/{k['set']}/{i}").random() < 0.5 else "B")
        if k["set"] in mx.EVAL_PICK:
            assert evr[i]["set"] == k["set"] and assign[i] == "search"
            assert i not in ("flan:000", "wmt24pp-0002", "wmt24pp-0004") and i not in blind_flan
        if k["set"] == "idiom":
            assert json.loads(evr[i]["meta"])["blind_set"] == "idiom" and i not in ("blind-003", "blind-006")
            assert k["blind_item"] == json.loads(evr[i]["meta"])["item"]
    novels = [k for k in key["pairs"] if k["set"] == "novels"]
    assert len({k["stratum"] for k in novels}) == 8 and "900" not in {k["stratum"] for k in novels}
    bt = [k["stratum"] for k in key["pairs"] if k["set"] == "bt"]
    assert sorted(bt) == sorted(SOURCES * 2)
    # the blind package: every differing new pair once, A/B as its key says
    inputs = [json.loads(x) for x in (out / "judge-input.jsonl").read_text().splitlines()]
    jkeys = [json.loads(x) for x in (out / "judge-key.jsonl").read_text().splitlines()]
    differing = {i for i in mbase if mx.norm(mbase[i]["hypothesis"]) != mx.norm(mown[i]["hypothesis"])}
    assert {k["item"] for k in jkeys} == differing and len(jkeys) == len(differing)
    assert [x["blind_id"] for x in inputs] == [k["blind_id"] for k in jkeys] == [f"rb-{n:03d}" for n in range(1, len(jkeys) + 1)]
    assert set(inputs[0]) == {"blind_id", "source", "A", "B"} and "set" not in inputs[0]
    for x, k in zip(inputs, jkeys):
        i = k["item"]
        assert x["source"] == mbase[i]["source"]
        assert x["A"] == (mbase if k["A"] == "base" else mown)[i]["hypothesis"]
        assert x["B"] == (mbase if k["B"] == "base" else mown)[i]["hypothesis"]
    in_sample = {(k["set"], k["item"]): k["n"] for k in key["pairs"] if k["set"] in ("novels", "bt")}
    assert {(k["set"], k["item"]): k["steekproef_n"] for k in jkeys if k["steekproef_n"]} == in_sample
    rep = json.loads((out / "select-report.json").read_text())
    assert rep["n"] == len(jkeys) and rep["in_steekproef"] == 18
    info = json.loads((out / "pick.json").read_text())
    assert info["counts"]["novels"]["differ_per_stratum"]["900"] == 0
    assert info["flags_new"]["G4-0"]["not_ended"] == 15
    printed = capsys.readouterr().out
    assert printed.startswith("MIXPICK ") and not any(r["hypothesis"] in printed for r in mbase.values())


def test_pick_stops_when_too_few_pairs_differ_or_the_split_is_not_g45s(tmp_path, monkeypatch):
    rng = random.Random(12)
    ev, assign, digest = eval_fixture(tmp_path, rng)
    mix_fixture(tmp_path, rng)
    with pytest.raises(SystemExit, match="not the split of G4-5.4"):
        mx.pick(pick_args(tmp_path))
    monkeypatch.setattr(g4arm, "SPLIT_DIGEST6C", digest)
    monkeypatch.setitem(mx.PICK, "ntrex", 20)
    with pytest.raises(SystemExit, match="ntrex: .* differing pairs in the search part, 20 wanted"):
        mx.pick(pick_args(tmp_path))


def test_named_takes_both_models_once():
    assert mx.named(["base=a", "G4-8-s140=b"], "--mix") == {"base": "a", "G4-8-s140": "b"}
    for bad in (["base=a"], ["base=a", "base=b"], ["base=a", "G4-7=b"]):
        with pytest.raises(SystemExit):
            mx.named(bad, "--mix")


# ---------------------------------------------------------------------------------------------------- the stage script

def test_the_translate_stage_uses_the_evaluations_own_translate_class():
    script = CODE / "run_g4mixsample.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    probe = (f'CODE={CODE}; eval "$(sed -n "/^TRANSLATE_CLASS=\'translate_class() {{\\$/,/^}}\'\\$/p" '
             f'"$CODE/run_g4arm.sh")"; printf %s "$TRANSLATE_CLASS"')
    got = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, check=True).stdout
    text = (CODE / "run_g4arm.sh").read_text()
    want = text[text.index("TRANSLATE_CLASS='") + len("TRANSLATE_CLASS='"):]
    want = want[:want.index("\n}'") + 2]
    assert got == want
    assert "--prompt natural-v1 --length-class $cls" in got and "translate.MODELS" in got
    body = script.read_text()
    assert 'eval "$(sed -n "/^TRANSLATE_CLASS=\'translate_class() {\\$/,/^}\'\\$/p" "$CODE/run_g4arm.sh")"' in body
    assert "rm -" not in body


def test_protocol_has_g4_8_15():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 19. G4-8 training"):text.index("## 20. Meting G4-judge-FB")]
    assert "*G4-8.15 Aanvulling 24 nov 2026" in section
    for n, what in ((8, "romans"), (10, "BT"), (6, "FLAN"), (5, "idioom")):
        assert re.search(rf"{what} \*\*{n}\*\*", section), what
    assert "FLORES **3**, NTREX **2**" in section
    assert sum(mx.PICK.values()) == 40 and mx.NOVELS_PER_BOOK * mx.NOVEL_BOOKS == 60 and mx.BT_PER_SOURCE * 5 == 60
