"""g4arm.py for arm G4-3 (protocol-g4.md §9, amendment G4-3) on small synthetic inputs (no tokenizer, no model, no
TPU): the packing of MaxText's SFT pipeline, the long validation rows, the set with all BT rows and replay apart per
pass, the check of distinct examples, the smoke test's verdict and estimate (three passes, or two above ~5 days),
the direct measure with the long rows beside the fixed reading, the classes of gemma4-31b-long."""

import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import g4arm  # noqa: E402
import g4lora  # noqa: E402

WORDS = "huis tafel stoel boom water vuur aarde lucht licht donker stad dorp weg brug rivier zee berg dal bos".split()
EN_WORDS = "house table chair tree water fire earth air light dark city village road bridge river sea hill dale".split()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(rng, words, n):
    return " ".join(rng.choice(words) for _ in range(n))


def write(rows, path, schema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def ns(**kw):
    return type("Args", (), kw)()


class FakeCounter:
    """Tokens = words (+60 for a prompt, +2 for a completion); lengths of plain texts = words"""

    def examples(self, rows):
        return [len(r["en"].split()) + 60 for r in rows], [len(r["nl"].split()) + 2 for r in rows]

    def lengths(self, texts):
        return [len(t.split()) for t in texts]


# ---------------------------------------------------------------------------------------------------- packing

def first_fit(lengths, rows, seq):
    """grain's PackAndBatchOperation written out: rows of a batch filled first-fit, a misfit opens the next batch"""
    batches = []
    cur = None
    for n in lengths:
        if cur is None:
            cur = [0] * rows
        for i in range(rows):
            if cur[i] + n <= seq:
                cur[i] += n
                break
        else:
            batches.append(cur)
            cur = [0] * rows
            cur[0] = n
    if cur is not None:
        batches.append(cur)
    return len(batches)


def test_pack_batches_is_first_fit_per_host_on_every_fourth_example():
    rng = random.Random(1)
    lengths = [rng.choice([3900, 2100, 800, 300, 120]) for _ in range(2000)]
    got = g4arm.pack_batches(lengths)
    assert got == [first_fit(lengths[h::4], 4, 4096) for h in range(4)]
    # four of the longest per batch: each a row of its own
    assert g4arm.pack_batches([3900] * 64) == [4, 4, 4, 4]
    # a short example after the long ones goes into the first row with room
    assert g4arm.pack_batches([3900, 3900, 3900, 3900, 150], hosts=1, rows=4) == [1]
    assert g4arm.pack_batches([3900] * 5, hosts=1, rows=4) == [2]
    with pytest.raises(SystemExit, match="sequence length"):
        g4arm.pack_batches([4097])
    assert g4arm.steps_of([1000, 1003, 998, 1001]) == 998 - 10


# ---------------------------------------------------------------------------------------------------- the datasets

def bt_rows(rng, source, split, n_short, n_long):
    rows = []
    for k in range(n_short + n_long):
        long_ = k >= n_short
        n = rng.randint(600, 1500) if long_ else rng.randint(15, 40)
        row = {"id": f"{source}:{split[0]}{k}", "nl": text(rng, WORDS, round(n * 1.1)), "en": text(rng, EN_WORDS, n),
               "hard_fails": [], "chrf": 70.0, "model": "google/gemma-4-31B-it@842da37",
               "tokens_nl": round(n * 1.1) + (0 if long_ else 0)}
        if long_:
            row["tokens_nl"] = max(row["tokens_nl"], 513)
        else:
            row["tokens_nl"] = min(row["tokens_nl"], 512)
        if source == "fineweb":
            row["dump"] = "CC-MAIN-2023-23"
        rows.append(row)
    return rows


def write_split(directory, split, rows_by_source):
    revisions = {}
    for source, (repo, revision) in g4arm.BT_REPOS3.items():
        f = directory / source / "data" / f"{split}-00000-of-00001.parquet"
        write(rows_by_source[source], f)
        revisions[source] = {"repo": repo, "revision": revision, "files": {f.name: sha(f)}}
    (directory / "revisions.json").write_text(json.dumps(revisions))
    return revisions


BOOKS = {"11-21": 700, "12-22": 300, "13-23": 250, "14-24": 120}


def g41_set(directory, rng):
    pairs = []
    for key, n in BOOKS.items():
        for k in range(n):
            en = text(rng, EN_WORDS, rng.randint(20, 60))
            pairs.append(g4arm.example({"pid": f"books:{key}-b1-{k}", "source": "books", "en": en,
                                        "nl": text(rng, WORDS, len(en.split())), "tokens_prompt": 80,
                                        "tokens_target": 40}, "book"))
    for k in range(200):
        en = text(rng, EN_WORDS, 30)
        pairs.append(g4arm.example({"pid": f"raad:t{k}", "source": "raad", "en": en, "nl": text(rng, WORDS, 30),
                                    "tokens_prompt": 90, "tokens_target": 32, "chrf": 70.0}, "bt"))
    d = Path(directory)
    meta_schema = pa.schema([(n, getattr(pa, t)()) for n, t in g4arm.META_SCHEMA_FIELDS])
    write([m for m, _ in pairs], d / "train.parquet")
    write([x for _, x in pairs], d / "train-meta.parquet", meta_schema)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({"set": "G4-1", "examples": len(pairs)}))


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = random.Random(7)
    val = {s: bt_rows(rng, s, "validation", 40, 12) for s in g4arm.BT_REPOS3}
    val["raad"][41]["hard_fails"] = ["loop"]
    write_split(tmp_path / "bt3-val", "validation", val)
    train = {s: bt_rows(rng, s, "train", 300, 30) for s in g4arm.BT_REPOS3}
    train["dpc"][0]["hard_fails"] = ["language"]
    train["dpc"][305]["hard_fails"] = ["loop"]
    write_split(tmp_path / "bt3", "train", train)
    # G4-2's held-out set: 800 BT validation rows (here: the short ones, repeated ids allowed by a smaller constant)
    monkeypatch.setattr(g4arm, "HELDOUT_BTVAL", 20)
    held = []
    for r in val["handelingen"][:20]:
        held.append({"id": f"heldout:btval:{r['id']}", "set": "btval", "source": r["en"], "reference": r["nl"],
                     "tokens": 30, "over_512": False, "length_class": "small",
                     "meta": json.dumps({"source": "handelingen", "pid": r["id"], "tokens_nl": 33})})
    held.append({"id": "heldout:books:books:12-22-b1-0", "set": "books", "source": "e", "reference": "n", "tokens": 1,
                 "over_512": False, "length_class": "tiny", "meta": json.dumps({"source": "books:12-22", "pid": "x"})})
    write(held, tmp_path / "heldout" / "evalsets.parquet", g4arm.schema(g4arm.EVAL_SCHEMA))
    (tmp_path / "heldout" / "books.json").write_text(json.dumps({"books": {"12-22": 300, "13-23": 250}, "pairs": 550}))
    g41_set(tmp_path / "G4-1", rng)
    write([{"id": "wmt-0", "set": "wmt24pp", "source": "a b c", "reference": "d e f", "meta": "{}"}],
          tmp_path / "evalsets.parquet")
    blind = tmp_path / "blind100.jsonl"
    blind.write_text(json.dumps({"blind_id": "blind-000", "source": "z y"}) + "\n")
    monkeypatch.setattr(g4arm, "blind_sha256", lambda: sha(blind))
    rng2 = random.Random(9)
    write([{"id": 10_000 + k, "en": text(rng2, EN_WORDS, 40), "nl": text(rng2, WORDS, 40), "task": "flan/t/x",
            "model": g4arm.REPLAY_MODEL2} for k in range(3000)], tmp_path / "replay.parquet")
    write([{"segments": [10_000, 10_001]}], tmp_path / "eval-rows.parquet")
    return {"tmp": tmp_path, "val": val, "train": train, "blind": blind}


def run_heldout3(world):
    out = world["tmp"] / "heldout3"
    g4arm.heldout3(ns(btval=str(world["tmp"] / "bt3-val"), heldout=str(world["tmp"] / "heldout" / "evalsets.parquet"),
                      evalsets=str(world["tmp"] / "evalsets.parquet"), blind=str(world["blind"]), tokenizer="unused",
                      workers=4, out=str(out)), counter=FakeCounter())
    return out


def test_heldout3_takes_the_long_validation_rows_in_whole_classes(world):
    import translate

    out = run_heldout3(world)
    m = json.loads((out / "manifest.json").read_text())
    rows = pq.read_table(out / "evalsets.parquet").to_pylist()
    assert m["rows"] == len(rows) and {r["set"] for r in rows} == {"btlong"}
    pids = {json.loads(r["meta"])["pid"] for r in rows}
    assert "raad:v41" not in pids                                   # hard_fails
    assert m["funnel"]["raad"]["hard_fails"] == 1
    assert not any(p.split(":")[1].startswith("v") and int(p.split(":v")[1]) < 40 for p in pids)  # only long rows
    classes = translate.MODELS[g4arm.LONG_MODEL]["length_classes"]
    for r in rows:
        c = classes[r["length_class"]]
        assert r["tokens"] <= c["max_source"]
        assert json.loads(r["meta"])["tokens_nl"] <= g4arm.DECODE_FIT * c["max_decode"]
    seen = []
    for k in range(4):
        seg = pq.read_table(out / f"segments-w{k}.parquet").to_pylist()
        seen += [s["id"] for s in seg]
        assert {r["length_class"] for r in rows if r["id"] in {s["id"] for s in seg}} == \
            set((out / f"classes-w{k}.txt").read_text().split())
    assert sorted(seen) == sorted(r["id"] for r in rows)
    ev = pq.read_table(out / "eval-btlong.parquet").column("messages").to_pylist()
    assert len(ev) == len(rows) and ev[0][0]["content"] == g4lora.user_turn(rows[0]["source"])
    assert m["btval_checked"] == 20


def test_heldout3_stops_when_g4_2s_validation_row_changed(world):
    f = world["tmp"] / "bt3-val" / "handelingen" / "data" / "validation-00000-of-00001.parquet"
    rows = pq.read_table(f).to_pylist()
    rows[3]["nl"] = rows[3]["nl"] + " extra"
    write(rows, f)
    rev = json.loads((world["tmp"] / "bt3-val" / "revisions.json").read_text())
    rev["handelingen"]["files"][f.name] = sha(f)
    (world["tmp"] / "bt3-val" / "revisions.json").write_text(json.dumps(rev))
    with pytest.raises(SystemExit, match="differs"):
        run_heldout3(world)


def test_revisions3_refuses_another_revision(world):
    rev = json.loads((world["tmp"] / "bt3" / "revisions.json").read_text())
    rev["handelingen"]["revision"] = "d61e32a7b9dedc7a38f800ad561bdd39168acf19"
    (world["tmp"] / "bt3" / "revisions.json").write_text(json.dumps(rev))
    with pytest.raises(SystemExit, match="pins"):
        g4arm.revisions3(world["tmp"] / "bt3")


def run_build3(world, heldout3_dir, **kw):
    out = world["tmp"] / "set3"
    args = dict(bt=str(world["tmp"] / "bt3"), base_set=str(world["tmp"] / "G4-1"),
                heldout_books=str(world["tmp"] / "heldout" / "books.json"),
                heldout=[str(world["tmp"] / "heldout" / "evalsets.parquet"), str(heldout3_dir / "evalsets.parquet")],
                replay=str(world["tmp"] / "replay.parquet"), replay_model=g4arm.REPLAY_MODEL2, replay_share=0.1,
                epochs=3, flan_eval=str(world["tmp"] / "eval-rows.parquet"),
                evalsets=str(world["tmp"] / "evalsets.parquet"), blind=str(world["blind"]), tokenizer="unused",
                out=str(out))
    args.update(kw)
    g4arm.build3(ns(**args), counter=FakeCounter())
    return out


def test_build3_takes_every_bt_row_without_hard_fails_the_books_and_replay_apart_per_pass(world):
    out = run_build3(world, run_heldout3(world))
    metas = pq.read_table(out / "G4-3" / "train-meta.parquet").to_pylist()
    msgs = pq.read_table(out / "G4-3" / "train.parquet").column("messages").to_pylist()
    report = json.loads((out / "report.json").read_text())
    n_bt = 5 * 330 - 2
    n_books = sum(BOOKS.values()) - 550
    n_replay = round((n_bt + n_books) * 0.1 / 0.9)
    assert len(metas) == len(msgs) == 3 * (n_bt + n_books + n_replay)
    first = [m for m in metas if m["epoch"] == 1]
    assert sum(m["kind"] == "bt" for m in first) == n_bt
    assert sum(m["kind"] == "bt" and m["long"] for m in first) == 5 * 30 - 1
    assert sum(m["kind"] == "book" for m in first) == n_books
    assert sum(m["kind"] == "replay" for m in first) == n_replay
    assert abs(n_replay / len(first) - 0.1) < 0.001
    assert "bt:dpc:t0" not in {m["id"] for m in metas} and "bt:dpc:t305" not in {m["id"] for m in metas}
    assert not any(m["kind"] == "book" and g4arm.book_key(m["id"]) in ("12-22", "13-23") for m in metas)
    # the replay differs in every pass, the rest is the same set in another order
    rep = [{m["id"] for m in metas if m["epoch"] == e and m["kind"] == "replay"} for e in (1, 2, 3)]
    assert not (rep[0] & rep[1]) and not (rep[1] & rep[2]) and not (rep[0] & rep[2])
    assert report["replay"]["apart_in_every_pass"] and report["replay"]["distinct"] == 3 * n_replay
    rest = [[m["id"] for m in metas if m["epoch"] == e and m["kind"] != "replay"] for e in (1, 2, 3)]
    assert sorted(rest[0]) == sorted(rest[1]) == sorted(rest[2]) and rest[0] != rest[1]
    assert [m["epoch"] for m in metas] == sorted(m["epoch"] for m in metas)
    # the estimate is the packing of the file's rows; the two-pass one of its first rows
    lengths = [m["tokens_prompt"] + m["tokens_target"] for m in metas]
    est = report["estimate"]
    assert est["batches_per_process"] == g4arm.pack_batches(lengths)
    assert est["two_passes"]["examples"] == sum(m["epoch"] <= 2 for m in metas)
    assert est["two_passes"]["batches_per_process"] == g4arm.pack_batches(lengths[:est["two_passes"]["examples"]])
    manifest = json.loads((out / "G4-3" / "manifest.json").read_text())
    assert manifest["seq"] == g4arm.SEQ3 and manifest["steps"] == est["steps"]
    assert manifest["inputs"]["bt"]["handelingen"]["revision"] == g4arm.BT_REPOS3["handelingen"][1]
    # the smoke set: pass 1's longest first, then others
    sm = pq.read_table(out / "G4-3-smoke" / "train-meta.parquet").to_pylist()
    sman = json.loads((out / "G4-3-smoke" / "manifest.json").read_text())
    n_long = min(g4arm.SMOKE3_LONGEST, len(first))
    tok = [m["tokens_prompt"] + m["tokens_target"] for m in sm]
    assert tok[:n_long] == sorted(tok[:n_long], reverse=True)
    assert min(tok[:n_long]) >= max(tok[n_long:], default=0)
    assert sman["batches_per_process"] == g4arm.pack_batches(tok)
    assert sman["min_of_longest"] == min(tok[:n_long])


def test_build3_stops_on_a_held_out_train_row_and_on_a_row_above_the_sequence(world, monkeypatch):
    h3 = run_heldout3(world)
    f = world["tmp"] / "bt3" / "raad" / "data" / "train-00000-of-00001.parquet"
    rows = pq.read_table(f).to_pylist()
    rows[2]["id"] = "raad:v45"                                      # a held-out long validation row
    write(rows, f)
    rev = json.loads((world["tmp"] / "bt3" / "revisions.json").read_text())
    rev["raad"]["files"][f.name] = sha(f)
    (world["tmp"] / "bt3" / "revisions.json").write_text(json.dumps(rev))
    held = {json.loads(r["meta"])["pid"] for r in pq.read_table(h3 / "evalsets.parquet").to_pylist()}
    assert "raad:v45" in held
    with pytest.raises(SystemExit, match="held-out"):
        run_build3(world, h3)
    monkeypatch.setattr(g4arm, "SEQ3", 1200)
    rows[2]["id"] = "raad:t2"
    write(rows, f)
    rev["raad"]["files"][f.name] = sha(f)
    (world["tmp"] / "bt3" / "revisions.json").write_text(json.dumps(rev))
    with pytest.raises(SystemExit, match="sequence length"):
        run_build3(world, h3)


def test_build3_reuses_replay_rows_only_where_the_pool_is_too_small(world):
    h3 = run_heldout3(world)
    rng = random.Random(4)
    write([{"id": 10_000 + k, "en": text(rng, EN_WORDS, 40), "nl": text(rng, WORDS, 40), "task": "flan/t/x",
            "model": g4arm.REPLAY_MODEL2} for k in range(300)], world["tmp"] / "replay.parquet")
    out = run_build3(world, h3)
    r = json.loads((out / "report.json").read_text())["replay"]
    assert not r["apart_in_every_pass"] and r["distinct"] == r["candidates"] < 3 * r["per_pass"]


# ---------------------------------------------------------------------------------------------------- check

def test_check3_checks_each_distinct_example_once_and_every_rows_count(world, monkeypatch):
    out = run_build3(world, run_heldout3(world))
    d = out / "G4-3"
    metas = pq.read_table(d / "train-meta.parquet").to_pylist()
    want = {m["id"]: m["tokens_prompt"] + m["tokens_target"] for m in metas}
    calls = []

    def fake(messages, ens, path, model, labels):
        calls.append(len(labels))
        return [want[i] for i in labels], sum(want[i] for i in labels), 0

    monkeypatch.setattr(g4lora, "check_examples", fake)
    g4arm.check(ns(data=str(d), tokenizer="unused", arm="G4-3", processes=1, chunk=500))
    assert sum(calls) == len(want) < len(metas)
    bad = dict(want)
    bad[next(iter(bad))] += 1
    monkeypatch.setattr(g4lora, "check_examples",
                        lambda m, e, p, mo, labels: ([bad[i] for i in labels], 0, 0))
    with pytest.raises(SystemExit, match="differs"):
        g4arm.check(ns(data=str(d), tokenizer="unused", arm="G4-3", processes=1, chunk=500))


# ---------------------------------------------------------------------------------------------------- smoke

def smoke3_summary(own0=0.08, batches=(21, 21, 21, 20), step_s=10.6, per_step=60000):
    evals = []
    for step in (0, 4, 8):
        for name in g4arm.EVAL_SETS3:
            evals.append({"step": step, "set": name, "batches": 3, "tokens": 900,
                          "loss": (own0 if name == "own" else 2.0) + 0.01 * step, "seconds": 40.0 + step})
    return {"steps_done": 8, "loss": [[k, 2.0 - 0.01 * k] for k in range(1, 9)], "peak_gib_max": 26.1,
            "lr_at": {"0": 0.0, "1": 1e-5, "7": 1e-6}, "eval": evals, "config": {"max_target_length": 4096},
            "epoch": {"batches_per_process": list(batches)}, "tokens": {"per_step": [per_step] * 8},
            "throughput": {"step_s_median": step_s}, "times": {"python_to_setup_s": 300.0, "setup_s": 125.0,
                                                               "compile_s_est": 420.0}}


SMOKE_MANIFEST = {"batches_per_process": [21, 21, 21, 20], "min_of_longest": 3700, "tokens": 1_500_000}
ADAPTER = {"rank": 32, "alpha": 16.0, "scale": 0.5}


def test_the_smoke3_verdict_needs_the_long_rows_the_packing_and_the_own_loss():
    assert g4arm.smoke3_verdict(smoke3_summary(), ADAPTER, SMOKE_MANIFEST)["passed"]
    v = g4arm.smoke3_verdict(smoke3_summary(own0=1.3), ADAPTER, SMOKE_MANIFEST)
    assert not v["passed"] and not v["checks"]["own_loss_step0"]["holds"]
    v = g4arm.smoke3_verdict(smoke3_summary(batches=(21, 21, 20, 21)), ADAPTER, SMOKE_MANIFEST)
    assert not v["checks"]["batches_as_build3"]["holds"]
    v = g4arm.smoke3_verdict(smoke3_summary(per_step=30000), ADAPTER, SMOKE_MANIFEST)
    assert not v["checks"]["longest_rows"]["holds"]
    s = smoke3_summary()
    s["eval"] = [e for e in s["eval"] if not (e["set"] == "btlong" and e["step"] == 4)]
    assert not g4arm.smoke3_verdict(s, ADAPTER, SMOKE_MANIFEST)["checks"]["eval_btlong"]["holds"]
    s = smoke3_summary()
    s["lr_at"] = {"0": 1e-4}
    assert not g4arm.smoke3_verdict(s, ADAPTER, SMOKE_MANIFEST)["checks"]["lr_peak"]["holds"]
    s = smoke3_summary()
    s["config"]["max_target_length"] = 2048
    assert not g4arm.smoke3_verdict(s, ADAPTER, SMOKE_MANIFEST)["checks"]["seq"]["holds"]


def test_the_estimate_takes_three_passes_up_to_five_days_and_two_above():
    report = {"estimate": {"steps": 8000, "tokens": 480_000_000, "fill": 0.92,
                           "two_passes": {"steps": 5333, "tokens": 320_000_000}}}
    x = g4arm.smoke3_estimate(smoke3_summary(), report, SMOKE_MANIFEST)
    three = x["three"]
    assert x["passes"] == 3 and x["steps"] == 8000
    assert three["evals"] == 8000 // g4arm.EVAL_EVERY3 + 2
    assert three["eval_round_s"] == 4 * 48.0
    want = 300 + 6.6 * (480_000_000 - 1_500_000) / 1e6 + 125 + 420 + 8000 * 10.6 + three["evals"] * 192.0
    assert three["seconds"] == round(want)
    # about 79 hours: above the old bound of 3 days, within 5 days: still three passes
    slow = g4arm.smoke3_estimate(smoke3_summary(step_s=34.0), report, SMOKE_MANIFEST)
    assert 72 < slow["three"]["hours"] <= 120 and slow["passes"] == 3
    slower = g4arm.smoke3_estimate(smoke3_summary(step_s=60.0), report, SMOKE_MANIFEST)
    assert slower["three"]["hours"] > 120 and slower["passes"] == 2 and slower["steps"] == 5333
    assert slower["hours"] == slower["two"]["hours"]


def test_smoke3_writes_the_passes_and_fails_only_on_a_check(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps(smoke3_summary(step_s=60.0)))
    (tmp_path / "a.json").write_text(json.dumps(ADAPTER))
    (tmp_path / "m.json").write_text(json.dumps(SMOKE_MANIFEST))
    (tmp_path / "r.json").write_text(json.dumps({"estimate": {"steps": 8000, "tokens": 480_000_000, "fill": 0.9,
                                                              "two_passes": {"steps": 5333, "tokens": 320_000_000}}}))
    args = ns(summary=str(tmp_path / "s.json"), adapter=str(tmp_path / "a.json"), smoke_manifest=str(tmp_path / "m.json"),
              data_report=str(tmp_path / "r.json"), out=str(tmp_path / "smoke.md"))
    g4arm.smoke3(args)
    r = json.loads((tmp_path / "smoke.json").read_text())
    assert r["passed"] and r["estimate"]["passes"] == 2
    assert "2 doorgangen" in (tmp_path / "smoke.md").read_text()
    (tmp_path / "s.json").write_text(json.dumps(smoke3_summary(own0=2.0)))
    with pytest.raises(SystemExit) as stop:
        g4arm.smoke3(args)
    assert stop.value.code == 1


# ---------------------------------------------------------------------------------------------------- the measure

def test_the_long_rows_are_reported_beside_the_fixed_reading(monkeypatch):
    import armeval

    monkeypatch.setattr(armeval, "chrf", lambda h, r: 50.0 if h == r else 10.0)
    monkeypatch.setattr(armeval, "corpus_chrf", lambda hs, rs: sum(50.0 if h == r else 10.0 for h, r in zip(hs, rs)) / len(hs))
    rows = [{"id": f"heldout:{s}:{k}", "set": s, "meta": json.dumps({"source": "raad" if s != "books" else "books:a-b"})}
            for s in ("books", "btval", "btlong") for k in range(3)]
    good = {r["id"]: {"id": r["id"], "hypothesis": "a", "reference": "a", "ended": True, "decode_loop": False,
                      "empty": False} for r in rows}
    bad = {i: dict(v, hypothesis="b") for i, v in good.items()}
    evals = [{"step": st, "set": s, "loss": x} for st, x in ((0, 2.0), (750, 1.0)) for s in g4arm.EVAL_SETS3]
    # worse on the long rows only: the reading of books and btval stays "closer"
    arm = {i: (bad[i] if "btlong" in i else good[i]) for i in good}
    h = g4arm.heldout_measure(rows, arm, bad, evals, "G4-3", "G4-0")
    assert h["verdict"] == "dichter bij het originele Nederlands"
    assert set(h["reading"]) == {"books", "btval"}
    assert h["reported"]["btlong"]["loss_down"] and not h["reported"]["btlong"]["chrf_up"]
    assert h["sets"]["btlong:raad"]["G4-3"]["n"] == 3
    md = "\n".join(g4arm.heldout_text(h, "G4-3", "G4-0"))
    assert "btlong (rapporteren, buiten de lezing)" in md


def test_gemma4_31b_long_holds_the_long_rows():
    import translate

    m = translate.MODELS[g4arm.LONG_MODEL]
    assert m["turn"] == translate.MODELS["gemma4-31b"]["turn"] and m["eos_ids"] == translate.MODELS["gemma4-31b"]["eos_ids"]
    classes = list(m["length_classes"].values())
    assert [c["max_source"] for c in classes] == sorted(c["max_source"] for c in classes)
    assert classes[-1]["max_source"] >= 2048 and g4arm.DECODE_FIT * classes[-1]["max_decode"] >= 2048
    for name, c in m["length_classes"].items():
        assert c["prefill_lengths"] == [c["max_source"] + 128]
        assert m["slots"][name] * (c["prefill_lengths"][-1] + c["max_decode"]) <= 16 * 2432   # d1k's slot-tokens
    assert g4arm.long_class(600, 700) == "e768"
    assert g4arm.long_class(600, 1300) is None                    # 1300 > 0.8 x 1536
    assert g4arm.long_class(1900, 2048) == "e2k"
    assert g4arm.long_class(3000, 100) is None
