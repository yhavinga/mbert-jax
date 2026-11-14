"""g4arm.py for arm G4-2 (protocol-g4.md §9, amendment G4-2) and sft_lora.py's held-out evals, on small synthetic
inputs (no tokenizer, no model, no TPU): the held-out books and BT validation, G4-0's own outputs as an eval file, the
FLAN replay candidates, the set with three passes, the smoke test's verdict, the direct measure's reading, the
template of training and translation."""

import hashlib
import json
import math
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
import sft_lora  # noqa: E402

WORDS = "huis tafel stoel boom water vuur aarde lucht licht donker stad dorp weg brug rivier zee berg dal bos".split()
EN_WORDS = "house table chair tree water fire earth air light dark city village road bridge river sea hill dale".split()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(rng, words, n):
    return " ".join(rng.choice(words) for _ in range(n))


def write(rows, path, schema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


class FakeCounter:
    """Tokens = words (+60 for a prompt, +2 for a completion); lengths of plain texts = words"""

    def examples(self, rows):
        return [len(r["en"].split()) + 60 for r in rows], [len(r["nl"].split()) + 2 for r in rows]

    def lengths(self, texts):
        return [len(t.split()) for t in texts]


BOOKS = {"11-21": 700, "12-22": 300, "13-23": 250, "14-24": 120, "15-25": 90, "16-26": 400}


def g41_set(directory, rng):
    """A G4-1 set: book pairs of six books, BT examples, as g4arm.build writes them"""
    pairs = []
    for key, n in BOOKS.items():
        for k in range(n):
            en = text(rng, EN_WORDS, rng.randint(20, 60))
            pairs.append((g4arm.example({"pid": f"books:{key}-b1-{k}", "source": "books", "en": en,
                                         "nl": text(rng, WORDS, len(en.split())), "tokens_prompt": 80,
                                         "tokens_target": 40}, "book")))
    for k in range(500):
        en = text(rng, EN_WORDS, 30)
        pairs.append(g4arm.example({"pid": f"raad:t{k}", "source": "raad", "en": en, "nl": text(rng, WORDS, 30),
                                    "tokens_prompt": 90, "tokens_target": 32, "chrf": 70.0}, "bt"))
    rng.shuffle(pairs)
    d = Path(directory)
    meta_schema = pa.schema([(n, getattr(pa, t)()) for n, t in g4arm.META_SCHEMA_FIELDS])
    write([m for m, _ in pairs], d / "train.parquet")
    write([x for _, x in pairs], d / "train-meta.parquet", meta_schema)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({"set": "G4-1", "examples": len(pairs)}))
    return pairs


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = random.Random(3)
    g41 = tmp_path / "G4-1"
    g41_set(g41, rng)
    btval = tmp_path / "btval"
    revisions = {}
    for source, (repo, revision) in g4arm.BT_REPOS.items():
        rows = []
        for k in range(300):
            n = rng.randint(15, 40)
            row = {"id": f"{source}:v{k}", "nl": text(rng, WORDS, round(n * 1.05)), "en": text(rng, EN_WORDS, n),
                   "hard_fails": [], "chrf": 60.0, "model": "google/gemma-4-31B-it@842da37"}
            if source == "fineweb":
                row["dump"] = "CC-MAIN-2023-23"
            rows.append(row)
        rows[0]["hard_fails"] = ["loop"]
        rows[1]["en"] = text(rng, EN_WORDS, 600)  # above class xlong
        rows[1]["nl"] = text(rng, WORDS, 600)
        f = btval / source / "data" / "validation-00000-of-00001.parquet"
        write(rows, f)
        revisions[source] = {"repo": repo, "revision": revision, "files": {f.name: sha(f)}}
    (btval / "revisions.json").write_text(json.dumps(revisions))
    bounds = {s: {"median": 1.0, "lo": 0.5, "hi": 2.0} for s in g4arm.BT_REPOS}
    (tmp_path / "data-report.json").write_text(json.dumps({"bt": {"length_bounds": bounds}}))
    write([{"id": "wmt-0", "set": "wmt24pp", "source": "a b c", "reference": "d e f", "meta": "{}"}],
          tmp_path / "evalsets.parquet")
    blind = tmp_path / "blind100.jsonl"
    blind.write_text(json.dumps({"blind_id": "blind-000", "source": "z y"}) + "\n")
    monkeypatch.setattr(g4arm, "blind_sha256", lambda: sha(blind))
    return {"tmp": tmp_path, "g41": g41, "btval": btval, "blind": blind, "rng": rng}


def ns(**kw):
    return type("Args", (), kw)()


def run_heldout(world):
    out = world["tmp"] / "heldout"
    g4arm.heldout(ns(g41=str(world["g41"]), btval=str(world["btval"]), data_report=str(world["tmp"] / "data-report.json"),
                     evalsets=str(world["tmp"] / "evalsets.parquet"), blind=str(world["blind"]), tokenizer="unused",
                     workers=4, out=str(out)), counter=FakeCounter())
    return out


def test_book_key_and_the_choice_of_whole_books():
    assert g4arm.book_key("book:books:74768-137379-b1-950") == "74768-137379"
    with pytest.raises(SystemExit):
        g4arm.book_key("bt:raad:x")
    chosen = g4arm.choose_heldout_books(BOOKS)
    assert "11-21" not in chosen                      # more than 600 pairs: never held out
    assert sum(chosen.values()) >= 500 and len(chosen) >= 2
    assert chosen == g4arm.choose_heldout_books(BOOKS)
    with pytest.raises(SystemExit):
        g4arm.choose_heldout_books({"a-b": 100, "c-d": 700})


def test_decode_class_needs_the_dutch_to_fit_the_limit():
    assert g4arm.decode_class(30, 50) == "tiny"        # tiny: source <= 32, decode 96
    assert g4arm.decode_class(60, 100) == "small"      # small: source <= 128, decode 256
    assert g4arm.decode_class(60, 230) is None         # 230 > 0.8 * 256
    assert g4arm.decode_class(600, 100) is None        # above xlong


def test_heldout_takes_whole_books_and_the_bt_validation(world):
    out = run_heldout(world)
    m = json.loads((out / "manifest.json").read_text())
    books = json.loads((out / "books.json").read_text())["books"]
    assert books == m["heldout_books"] == g4arm.choose_heldout_books(BOOKS)
    rows = pq.read_table(out / "evalsets.parquet").to_pylist()
    by_set = {s: [r for r in rows if r["set"] == s] for s in ("books", "btval")}
    assert len(by_set["btval"]) == g4arm.HELDOUT_BTVAL
    assert 0 < len(by_set["books"]) <= g4arm.HELDOUT_BOOKS_SAMPLE
    assert {json.loads(r["meta"])["source"].split(":")[1] for r in by_set["books"]} <= set(books)
    # the bounds of the pool: hard_fails and a source above xlong never come in
    pids = {json.loads(r["meta"])["pid"] for r in by_set["btval"]}
    assert not pids & {f"{s}:v{k}" for s in g4arm.BT_REPOS for k in (0, 1)}
    assert m["btval_funnel"]["hard_fails"] == 5
    # segments per worker hold every row once, in whole classes
    seen = []
    for k in range(4):
        seg = pq.read_table(out / f"segments-w{k}.parquet").to_pylist()
        seen += [r["id"] for r in seg]
        classes = (out / f"classes-w{k}.txt").read_text().split()
        assert {r["length_class"] for r in rows if r["id"] in {s["id"] for s in seg}} == set(classes)
    assert sorted(seen) == sorted(r["id"] for r in rows)
    # eval files in the training format: natural-v1's user turn and the original Dutch
    ev = pq.read_table(out / "eval-btval.parquet").column("messages").to_pylist()
    assert len(ev) == g4arm.HELDOUT_BTVAL
    assert ev[0][0]["content"] == g4lora.user_turn(by_set["btval"][0]["source"])
    assert ev[0][1]["content"] == by_set["btval"][0]["reference"]
    run_heldout(world)
    assert (out / "evalsets.parquet").read_bytes() == (world["tmp"] / "heldout" / "evalsets.parquet").read_bytes()


def test_a_validation_row_that_is_a_training_row_stops_heldout(world):
    d = world["btval"] / "raad" / "data"
    f = d / "validation-00000-of-00001.parquet"
    rows = pq.read_table(f).to_pylist()
    rows[5]["id"] = "raad:t3"                          # G4-1 trained on bt:raad:t3
    write(rows, f)
    rev = json.loads((world["btval"] / "revisions.json").read_text())
    rev["raad"]["files"][f.name] = sha(f)
    (world["btval"] / "revisions.json").write_text(json.dumps(rev))
    with pytest.raises(SystemExit, match="training rows"):
        run_heldout(world)


def test_own_takes_g4_0s_usable_translations_of_the_bt_validation(world):
    out = run_heldout(world)
    rows = pq.read_table(out / "evalsets.parquet").to_pylist()
    pairs = [{"id": r["id"], "set": r["set"], "hypothesis": "eigen " + r["reference"], "ended": True,
              "decode_loop": False, "empty": False} for r in rows]
    pairs[[r["set"] for r in rows].index("btval")]["ended"] = False
    write(pairs, world["tmp"] / "own" / "pairs-w0.parquet")
    target = world["tmp"] / "own" / "eval-own.parquet"
    g4arm.own(ns(heldout=str(out / "evalsets.parquet"), pairs=str(world["tmp"] / "own" / "pairs-w*.parquet"),
                 tokenizer="unused", out=str(target)), counter=FakeCounter())
    msgs = pq.read_table(target).column("messages").to_pylist()
    assert len(msgs) == g4arm.HELDOUT_BTVAL - 1
    assert all(m[1]["content"].startswith("eigen ") for m in msgs)
    write(pairs[:10], world["tmp"] / "own" / "pairs-w0.parquet")
    with pytest.raises(SystemExit, match="without G4-0"):
        g4arm.own(ns(heldout=str(out / "evalsets.parquet"), pairs=str(world["tmp"] / "own" / "pairs-w*.parquet"),
                     tokenizer="unused", out=str(target)), counter=FakeCounter())


def flan_world(tmp, rng):
    """A tiny Gemma 4 FLAN-nl after its post step: plan2, shards, measures, assemble's kept examples, setup"""
    plan = tmp / "plan2"
    segs = []
    for k in range(400):
        segs.append({"id": k, "text": text(rng, EN_WORDS, rng.randint(5, 120)), "tokens": 10})
    write(segs[:200], plan / "segments-w0.parquet")
    write(segs[200:], plan / "segments-w1.parquet")
    (plan / "plan.json").write_text(json.dumps({"template_segments": {"Answer:": 399}}))
    examples = []
    for p in range(150):
        a, b = 2 * p, 2 * p + 1
        examples.append({"host": 0, "position": p, "source": f"flan/t/task{p % 7}",
                         "status": "translate" if p % 10 else "task:code",
                         "instruction": json.dumps([["T", "Answer:"], ["S", a], ["L", " "], ["S", 399]]),
                         "answer": json.dumps([["S", b]])})
    write(examples, plan / "examples.parquet")
    shards = tmp / "shards" / "small"
    write([{"id": s["id"], "text_nl": "nl " + s["text"], "ended": True} for s in segs], shards / "small-w0-0000.parquet")
    measures = []
    for s in segs:
        n = len(s["text"].split()) + 1
        measures.append({"id": s["id"], "check": "language" if s["id"] % 13 == 0 else "", "af": True,
                         "executed": s["id"] % 17 == 0, "chatter": False, "english_left": False,
                         "identical_to_source": False, "repairs": "", "tokens_out": n})
    write(measures, tmp / "report" / "measures.parquet")
    kept = [p for p in range(150) if p % 10 and p != 3]          # assemble dropped example 3
    write([{"position": p, "source": "x", "inputs": "i", "targets": "t"} for p in kept], tmp / "nl" / "w0.parquet")
    (tmp / "report" / "setup.json").write_text(json.dumps({"lines": {"Model": "Gemma 4 31B-it ...; prompt flan-v1"}}))
    return {"plan": plan, "shards": tmp / "shards", "measures": tmp / "report" / "measures.parquet", "nl": tmp / "nl",
            "setup": tmp / "report" / "setup.json", "segs": segs, "kept": kept}


URIS = [f"{k}=gs://t5-dutch-english-usc2{v}" for k, v in g4arm.REPLAY_URIS2.items()]


def run_replay(fw, out, uris=URIS):
    g4arm.replay_candidates(ns(plan=str(fw["plan"]), shards=str(fw["shards"]), measures=str(fw["measures"]),
                               nl=str(fw["nl"]), setup=str(fw["setup"]), out=str(out), uri=uris))


def test_replay_takes_segments_of_kept_examples_that_pass_the_measures(tmp_path):
    fw = flan_world(tmp_path, random.Random(5))
    out = tmp_path / "replay.parquet"
    run_replay(fw, out)
    rows = pq.read_table(out).to_pylist()
    ids = {r["id"] for r in rows}
    allowed = {2 * p for p in fw["kept"]} | {2 * p + 1 for p in fw["kept"]}
    assert ids and ids <= allowed                      # S pieces of kept examples only
    assert 399 not in ids                              # the template segment
    assert 6 not in ids and 7 not in ids               # example 3 was dropped by assemble
    assert not {i for i in ids if i % 13 == 0 or i % 17 == 0}
    lo, hi = g4arm.REPLAY_TOKENS2
    words = {s["id"]: len(s["text"].split()) + 1 for s in fw["segs"]}
    assert all(lo <= words[i] <= hi for i in ids)
    assert all("gemma-4" in r["model"] and r["nl"] == "nl " + r["en"] for r in rows)
    assert rows[0]["task"].startswith("flan/t/task")


def test_replay_refuses_another_source_than_the_gemma4_flan_nl(tmp_path):
    fw = flan_world(tmp_path, random.Random(5))
    bad = [u.replace("translations/g4", "translations/tg27b") for u in URIS]
    with pytest.raises(SystemExit, match="not the Gemma 4"):
        run_replay(fw, tmp_path / "r.parquet", bad)
    bad = [u.replace("/nl-g4", "/nl") for u in URIS]
    with pytest.raises(SystemExit, match="not the Gemma 4"):
        run_replay(fw, tmp_path / "r.parquet", bad)
    fw["setup"].write_text(json.dumps({"lines": {"Model": "TranslateGemma 27B"}}))
    with pytest.raises(SystemExit, match="setup"):
        run_replay(fw, tmp_path / "r.parquet")


def replay_file(path, n=3000, model=g4arm.REPLAY_MODEL2):
    rng = random.Random(9)
    write([{"id": 10_000 + k, "en": text(rng, EN_WORDS, 40), "nl": text(rng, WORDS, 40), "task": "flan/t/x",
            "model": model} for k in range(n)], path)


def run_build2(world, heldout_dir, replay, **kw):
    out = world["tmp"] / "set2"
    write([{"segments": [10_000, 10_001]}], world["tmp"] / "eval-rows.parquet")
    args = dict(base_set=str(world["g41"]), heldout_books=str(heldout_dir / "books.json"), replay=str(replay),
                replay_model=g4arm.REPLAY_MODEL2, replay_share=0.2, epochs=3,
                flan_eval=str(world["tmp"] / "eval-rows.parquet"), evalsets=str(world["tmp"] / "evalsets.parquet"),
                blind=str(world["blind"]), tokenizer="unused", out=str(out))
    args.update(kw)
    g4arm.build2(ns(**args), counter=FakeCounter())
    return out


def test_build2_is_g4_1_without_the_heldout_books_plus_20_percent_replay_in_three_passes(world):
    heldout_dir = run_heldout(world)
    replay = world["tmp"] / "replay.parquet"
    replay_file(replay)
    out = run_build2(world, heldout_dir, replay)
    metas = pq.read_table(out / "G4-2" / "train-meta.parquet").to_pylist()
    msgs = pq.read_table(out / "G4-2" / "train.parquet").column("messages").to_pylist()
    books = json.loads((heldout_dir / "books.json").read_text())["books"]
    n_g41 = sum(BOOKS.values()) + 500
    per_epoch = n_g41 - sum(books.values())
    n_replay = round(per_epoch * 0.25)
    assert len(metas) == len(msgs) == 3 * (per_epoch + n_replay)
    first = [m for m in metas if m["epoch"] == 1]
    assert sum(m["kind"] == "replay" for m in first) == n_replay
    assert abs(n_replay / len(first) - 0.2) < 0.001
    assert not any(m["kind"] == "book" and g4arm.book_key(m["id"]) in books for m in metas)
    assert "replay:10000" not in {m["id"] for m in metas}       # a segment of the FLAN evaluation rows
    ids = [[m["id"] for m in metas if m["epoch"] == e] for e in (1, 2, 3)]
    assert sorted(ids[0]) == sorted(ids[1]) == sorted(ids[2]) and ids[0] != ids[1] != ids[2]
    assert [m["epoch"] for m in metas] == sorted(m["epoch"] for m in metas)
    # the G4-1 examples are G4-1's, the replay in natural-v1 without an instruction sentence
    g41 = {m["id"]: x for x, m in zip(pq.read_table(world["g41"] / "train.parquet").column("messages").to_pylist(),
                                     pq.read_table(world["g41"] / "train-meta.parquet").to_pylist())}
    for x, m in zip(msgs, metas):
        if m["kind"] != "replay":
            assert x == g41[m["id"]]
        else:
            assert x[0]["content"] == g4lora.user_turn(m["en"])
    manifest = json.loads((out / "G4-2" / "manifest.json").read_text())
    assert manifest["epochs"] == 3 and manifest["examples_per_epoch"] == per_epoch + n_replay
    assert manifest["inputs"]["base_set"]["train.parquet"] == sha(world["g41"] / "train.parquet")
    assert (out / "G4-2" / "template.jinja").read_text() == g4lora.TEMPLATE


def test_build2_refuses_a_replay_that_is_not_gemma4(world):
    heldout_dir = run_heldout(world)
    replay = world["tmp"] / "replay.parquet"
    replay_file(replay, model="google/translategemma-27b-it")
    with pytest.raises(SystemExit, match="not Gemma 4"):
        run_build2(world, heldout_dir, replay)
    replay_file(replay)
    with pytest.raises(SystemExit, match="Gemma 4"):
        run_build2(world, heldout_dir, replay, replay_model="translategemma")


def smoke2_summary(own0=0.31):
    evals = []
    for step in (0, 4, 8):
        for name, loss in (("books", 2.4), ("btval", 2.1), ("own", own0)):
            evals.append({"step": step, "set": name, "batches": 3, "tokens": 900, "loss": loss - step * 0.01})
    return {"steps_done": 8, "loss": [[k, 2.3 - 0.01 * k] for k in range(1, 9)], "peak_gib_max": 24.4,
            "lr_at": {"0": 1e-4, "1": 1e-4, "7": 1e-4}, "eval": evals}


def test_the_smoke2_verdict_needs_every_check_and_the_own_loss_gate():
    adapter = {"rank": 32, "alpha": 16.0, "scale": 0.5}
    assert g4arm.smoke2_verdict(smoke2_summary(), adapter)["passed"]
    v = g4arm.smoke2_verdict(smoke2_summary(own0=1.2), adapter)
    assert not v["passed"] and not v["checks"]["own_loss_step0"]["holds"]
    s = smoke2_summary()
    s["eval"] = [e for e in s["eval"] if not (e["set"] == "btval" and e["step"] == 8)]
    assert not g4arm.smoke2_verdict(s, adapter)["checks"]["eval_btval"]["holds"]
    s = smoke2_summary()
    s["lr_at"] = {"0": 1e-5}
    assert not g4arm.smoke2_verdict(s, adapter)["checks"]["lr_peak"]["holds"]
    s = smoke2_summary()
    s["loss"][3][1] = math.nan
    assert not g4arm.smoke2_verdict(s, adapter)["passed"]


def test_the_direct_measure_reads_closer_not_closer_or_mixed(monkeypatch):
    import armeval

    # chrF without sacrebleu: the share of reference words in the hypothesis, x 100
    def share(h, r):
        return 100 * len(set(h.split()) & set(r.split())) / max(1, len(set(r.split())))

    monkeypatch.setattr(armeval, "chrf", share)
    monkeypatch.setattr(armeval, "corpus_chrf", lambda hs, rs: sum(map(share, hs, rs)) / len(hs))
    rows = [{"id": f"heldout:{s}:{k}", "set": s, "meta": json.dumps({"source": "raad" if s == "btval" else "books:a-b"})}
            for s in ("books", "btval") for k in range(3)]
    ref = {r["id"]: "a b c d" for r in rows}
    good = {i: {"id": i, "hypothesis": "a b c", "reference": ref[i], "ended": True, "decode_loop": False, "empty": False}
            for i in ref}
    bad = {i: dict(v, hypothesis="a x y") for i, v in good.items()}
    evals = [{"step": 0, "set": s, "loss": 2.0} for s in ("books", "btval")] + \
            [{"step": 300, "set": s, "loss": 1.5} for s in ("books", "btval")]
    h = g4arm.heldout_measure(rows, good, bad, evals, "G4-2", "G4-0")
    assert h["verdict"] == "dichter bij het originele Nederlands"
    assert h["sets"]["btval:raad"]["G4-2"]["n"] == 3
    worse = [dict(e, loss=2.5) if e["step"] == 300 else e for e in evals]
    assert g4arm.heldout_measure(rows, bad, good, worse, "G4-2", "G4-0")["verdict"] == "niet dichter"
    assert g4arm.heldout_measure(rows, bad, good, evals, "G4-2", "G4-0")["verdict"] == "gemengd"
    with pytest.raises(SystemExit):
        g4arm.heldout_measure(rows, {}, good, evals, "G4-2", "G4-0")


def test_the_training_prompt_is_translates_prompt_and_the_template_renders_it():
    """amendment G4-2.8: the user turn, the model's turn opened with the empty thought channel, the Dutch and <turn|>"""
    import jinja2
    import translate

    turn = translate.MODELS["gemma4-31b"]["turn"]
    template = jinja2.Environment().from_string(g4lora.TEMPLATE)
    for en, nl in (("The cat sat on the mat.", "De kat zat op de mat."), ("  Two\nlines ", "Twee\nregels"),
                   ("Answer: yes or no?", "Antwoord: ja of nee?")):
        prompt = turn["prefix"] + translate.user_text(translate.PROMPTS["natural-v1"], en, "en", "nl") + turn["suffix"]
        assert g4lora.prompt_text(en) == prompt
        assert prompt.endswith("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        full = template.render(messages=[{"role": "user", "content": g4lora.user_turn(en)},
                                         {"role": "assistant", "content": nl}], bos_token="<bos>",
                               add_generation_prompt=False)
        assert full == prompt + nl + "<turn|>\n"
    assert 106 in translate.MODELS["gemma4-31b"]["eos_ids"]        # <turn|> ends a translation


def test_sft_lora_eval_helpers():
    assert sft_lora.parse_evals(["books=/a.parquet", "own=/b.parquet"]) == {"books": "/a.parquet", "own": "/b.parquet"}
    with pytest.raises(SystemExit):
        sft_lora.parse_evals(["books"])
    with pytest.raises(SystemExit):
        sft_lora.parse_evals(["a=x", "a=y"])
    assert sft_lora.eval_overrides({}, 0) == []
    assert sft_lora.eval_overrides({"books": "/a", "own": "/b"}, 150) == [
        "eval_interval=150", "hf_eval_files=/a", "hf_eval_split=train", "eval_steps=-1"]
    with pytest.raises(SystemExit):
        sft_lora.eval_overrides({"books": "/a"}, 0)
    r = sft_lora.eval_loss("books", 150, [(30.0, 10.0), (0.0, 0.0), (20.0, 15.0)], 3, 12.34)
    assert r == {"step": 150, "set": "books", "batches": 3, "tokens": 25, "loss": 2.0, "seconds": 12.3}
    with pytest.raises(SystemExit):
        sft_lora.eval_loss("books", 0, [(1.0, 1.0)], 2, 0.0)
    with pytest.raises(SystemExit):
        sft_lora.eval_loss("books", 0, [(0.0, 0.0)], 1, 0.0)
    assert sft_lora.needs_final_eval([{"step": 1200}], 1351)
    assert not sft_lora.needs_final_eval([{"step": 1350}], 1350)
    assert not sft_lora.needs_final_eval([], 8)
    assert sft_lora.lr_points(1350, 0.03, 0.1) == [0, 1, 40, 41, 675, 1215, 1349]
    assert sft_lora.lr_points(8, 0.03, 0.1) == [0, 1, 4, 7]


def test_report_of_g4_2_names_the_arm_and_adds_the_direct_measure(tmp_path, monkeypatch):
    import armeval

    monkeypatch.setattr(g4arm, "N_WMT_GOOD", 2)
    monkeypatch.setattr(g4arm, "N_FLAN", 1)
    monkeypatch.setattr(armeval, "chrf", lambda h, r: 50.0 if h == r else 10.0)
    monkeypatch.setattr(armeval, "corpus_chrf", lambda hs, rs: sum(50.0 if h == r else 10.0 for h, r in zip(hs, rs)) / len(hs))
    evalsets = [{"id": f"{s}-{k}", "set": s, "source": "a b c", "reference": None if s in ("flan", "blind") else "x y z",
                 "tokens": 3, "over_512": False, "length_class": "tiny", "meta": "{}"}
                for s in ("wmt24pp", "flores", "ntrex", "europarl", "flan", "blind") for k in range(2)]
    evalsets = [r for r in evalsets if r["id"] != "flan-1"]
    write(evalsets, tmp_path / "evalsets.parquet")
    for name, qe in (("G4-2", 1.0), ("G4-0", 2.0)):
        d = tmp_path / name
        write([{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": "p q r", "reference": r["reference"],
                "ended": True, "decode_loop": False, "english_left": False, "empty": False} for r in evalsets],
              d / "pairs" / "pairs-w0.parquet")
        write([{"id": r["id"], "qe_source_hypothesis": qe,
                "ref_source_hypothesis_reference": qe if r["reference"] else None} for r in evalsets],
              d / "scores" / "metricx-w0.parquet")
        write([{"id": r["id"], "comet22": 0.85} for r in evalsets if r["reference"]], d / "scores" / "comet-w0.parquet")
    write([{"id": r["id"], "set": r["set"], "metricx_qe": 1.5, "metricx_ref": 1.4, "comet22": 0.86}
           for r in evalsets if r["set"] in g4arm.FOUR_SETS], tmp_path / "tg.parquet")
    preds = []
    for model, p in (("G4-2", 0.9), ("G4-0", 0.1), ("G4-1", 0.1), ("TG", 0.1)):
        preds += [{"model": model, "set": s, "id": i, "p_original": p, "truncated": False}
                  for s, i in (("wmt24pp", "wmt24pp-0"), ("wmt24pp", "wmt24pp-1"), ("flan", "flan-0"))]
    write(preds, tmp_path / "pred.parquet")
    held = [{"id": f"heldout:{s}:{k}", "set": s, "source": "e n", "reference": "o r", "tokens": 2, "over_512": False,
             "length_class": "tiny", "meta": json.dumps({"source": "raad" if s == "btval" else "books:a-b"})}
            for s in ("books", "btval") for k in range(2)]
    write(held, tmp_path / "held.parquet")
    for name, hyp in (("G4-2", "o r"), ("G4-0", "x")):
        write([{"id": r["id"], "set": r["set"], "hypothesis": hyp, "reference": r["reference"], "ended": True,
                "decode_loop": False, "empty": False} for r in held], tmp_path / "h" / name / "pairs-w0.parquet")
    summary = {"steps": 300, "steps_done": 300, "loss": [[1, 2.3], [300, 0.8]],
               "eval": [{"step": st, "set": s, "loss": x, "tokens": 9} for st, x in ((0, 2.2), (300, 1.4))
                        for s in ("books", "btval", "own")], "lr_at": {"0": 0.0, "100": 1e-4}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    out = tmp_path / "report.md"
    args = ns(evalsets=str(tmp_path / "evalsets.parquet"), arm=str(tmp_path / "G4-2"), base=str(tmp_path / "G4-0"),
              tg_scores=str(tmp_path / "tg.parquet"), predictions=str(tmp_path / "pred.parquet"), classifier_results=None,
              lora_line=[], train_summary=str(tmp_path / "summary.json"), tally=None, arm_name="G4-2",
              heldout=str(tmp_path / "held.parquet"), heldout_arm=str(tmp_path / "h" / "G4-2" / "pairs-w*.parquet"),
              heldout_base=str(tmp_path / "h" / "G4-0" / "pairs-w*.parquet"), out=str(out))
    g4arm.report(args)
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["arm"] == "G4-2" and r["b"]["arm"] == pytest.approx(1.0) and r["b"]["holds"]
    assert r["heldout"]["verdict"] == "dichter bij het originele Nederlands"
    assert r["heldout"]["reading"]["books"]["loss_step0"] == 2.2 and r["training"]["lr_at"]["100"] == 1e-4
    text = out.read_text()
    assert "# Arm G4-2" in text and "$S/blind/G4-2/" in text and "De directe maat" in text
