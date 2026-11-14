"""g4arm.py: the training set of arm G4-1 (protocol-g4.md §2), its evaluation segments, the rules of §5 and the smoke
test's verdict (§6), on small synthetic inputs (no tokenizer, no TPU)."""

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

WORDS = ("huis tafel stoel boom water vuur aarde lucht licht donker stad dorp weg brug rivier zee berg dal bos veld "
         "kat hond paard koe schaap geit vogel vis bloem gras").split()
EN_WORDS = ("house table chair tree water fire earth air light dark city village road bridge river sea mountain "
            "valley forest field cat dog horse cow sheep goat bird fish flower grass").split()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(rng, words, n):
    return " ".join(rng.choice(words) for _ in range(n))


class FakeCounter:
    """Tokens = words of the prompt's English + 60, and of the Dutch + 2"""

    def examples(self, rows):
        return [len(r["en"].split()) + 60 for r in rows], [len(r["nl"].split()) + 2 for r in rows]


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rng = random.Random(7)
    bt = tmp_path / "bt"
    revisions = {}
    special = {}
    for source, (repo, revision) in g4arm.BT_REPOS.items():
        rows = []
        for k in range(400):
            n_en = rng.randint(20, 40)
            n_nl = round(n_en * rng.uniform(0.85, 1.3))
            row = {"id": f"{source}:{k}", "nl": text(rng, WORDS, n_nl), "en": text(rng, EN_WORDS, n_en),
                   "hard_fails": [], "chrf": round(rng.uniform(50, 90), 3), "model": "google/gemma-4-31B-it@842da37"}
            if source == "fineweb":
                row["dump"] = f"CC-MAIN-{rng.choice([2019, 2022, 2023, 2024])}-{rng.randint(10, 50):02d}"
            rows.append(row)
        rows[0]["hard_fails"] = ["structure"]
        rows[1]["nl"] = text(rng, WORDS, 120)  # a ratio far above the source's median: the length bound
        # a 13-gram of an evaluation set's source, the Dutch as long as the English
        rows[2]["en"] = "the quick brown fox jumps over the lazy dog near the river bank today " + rows[2]["en"]
        rows[2]["nl"] = text(rng, WORDS, 14) + " " + rows[2]["nl"]
        rows[3]["en"] = text(rng, EN_WORDS, 1990)  # too long for 2,048 tokens
        rows[3]["nl"] = text(rng, WORDS, 2000)
        special[source] = [r["id"] for r in rows[:4]]
        f = bt / source / "data" / "train-00000-of-00001.parquet"
        write(rows, f)
        revisions[source] = {"repo": repo, "revision": revision, "files": {f.name: sha(f)}}
    (bt / "revisions.json").write_text(json.dumps(revisions))
    books = tmp_path / "rl"
    metas, msgs = [], []
    for k in range(300):
        en = text(rng, EN_WORDS, rng.randint(30, 60))
        nl = text(rng, WORDS, round(len(en.split()) * rng.uniform(0.9, 1.1)))
        metas.append({"id": f"real:books:{k}", "kind": "real", "source": "books", "en": en})
        msgs.append({"messages": [{"role": "user", "content": "chat-v1 ..." + en}, {"role": "assistant", "content": nl}]})
    for k in range(50):
        metas.append({"id": f"bt:handelingen:{k}", "kind": "bt", "source": "handelingen", "en": "x"})
        msgs.append({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]})
    write(metas, books / "train-meta.parquet")
    write(msgs, books / "train.parquet")
    r2 = tmp_path / "r2"
    metas, msgs = [], []
    for k in range(500):
        n_en = rng.randint(20, 40)
        metas.append({"id": f"real:{k}", "kind": "real", "source": "europarl", "en": text(rng, EN_WORDS, n_en)})
        msgs.append({"messages": [{"role": "user", "content": "u"},
                                  {"role": "assistant", "content": text(rng, WORDS, round(n_en * rng.uniform(0.9, 1.25)))}]})
    write(metas, r2 / "train-meta.parquet")
    write(msgs, r2 / "train.parquet")
    evalsets = [{"id": "wmt-0", "set": "wmt24pp", "source": "the quick brown fox jumps over the lazy dog near the river "
                 "bank today", "reference": "de snelle", "meta": "{}"},
                {"id": "flores-0", "set": "flores", "source": "a b", "reference": "c d", "meta": "{}"}]
    write(evalsets, tmp_path / "evalsets.parquet")
    blind = tmp_path / "blind100.jsonl"
    blind.write_text(json.dumps({"blind_id": "blind-000", "id": "x", "set": "idiom", "source": "z y"}) + "\n")
    monkeypatch.setattr(g4arm, "FOUR_SETS", {"wmt24pp": 1, "flores": 1})
    monkeypatch.setattr(g4arm, "BOOKS_EXPECTED", 300)
    monkeypatch.setattr(g4arm, "blind_sha256", lambda: sha(blind))
    return {"bt": bt, "books": books, "r2": r2, "evalsets": tmp_path / "evalsets.parquet", "blind": blind,
            "special": special, "tmp": tmp_path}


def args_for(inputs, out, **extra):
    base = {"bt": str(inputs["bt"]), "books": str(inputs["books"]), "match_set": str(inputs["r2"]),
            "evalsets": str(inputs["evalsets"]), "blind": str(inputs["blind"]), "flan_eval": None,
            "tokenizer": "unused", "replay_share": 0.0, "replay": None, "replay_model": None,
            "fineweb_max_crawl": None, "out": str(out)}
    base.update(extra)
    return type("Args", (), base)()


def test_build_draws_one_back_translation_per_book_pair_and_applies_every_bound(inputs):
    out = inputs["tmp"] / "out"
    g4arm.build(args_for(inputs, out), counter=FakeCounter())
    metas = pq.read_table(out / "G4-1" / "train-meta.parquet").to_pylist()
    msgs = pq.read_table(out / "G4-1" / "train.parquet").column("messages").to_pylist()
    kinds = [m["kind"] for m in metas]
    assert kinds.count("book") == 300 and kinds.count("bt") == 300 and "replay" not in kinds
    ids = {m["id"] for m in metas}
    for source, (hard, length, gram, long) in inputs["special"].items():
        assert f"bt:{hard}" not in ids and f"bt:{length}" not in ids and f"bt:{gram}" not in ids and f"bt:{long}" not in ids
    report = json.loads((out / "report.json").read_text())
    funnel = report["bt"]["funnel"]["dpc"]
    assert funnel["hard_fails"] == 1 and funnel["length"] >= 1 and funnel["eval_13gram"] == 1 and funnel["too_long"] == 1
    assert report["match"]["drawn"] == 300 and not report["match"]["limited"] and report["match"]["bins"] == 5
    for m, x in zip(metas, msgs):
        user, assistant = x
        assert user["content"] == g4lora.user_turn(m["en"])
        assert user["content"].startswith("Translate the following English text into natural, idiomatic Dutch")
        assert m["tokens_prompt"] + m["tokens_target"] <= g4arm.MAX_TOKENS
    assert (out / "G4-1" / "template.jinja").read_text() == g4lora.TEMPLATE
    manifest = json.loads((out / "G4-1" / "manifest.json").read_text())
    assert manifest["kinds"] == {"bt": 300, "book": 300} and manifest["replay_share"] == 0.0
    assert "| dpc |" in (out / "report.md").read_text()


def test_build_is_the_same_twice(inputs):
    a, b = inputs["tmp"] / "a", inputs["tmp"] / "b"
    g4arm.build(args_for(inputs, a), counter=FakeCounter())
    g4arm.build(args_for(inputs, b), counter=FakeCounter())
    assert pq.read_table(a / "G4-1" / "train.parquet").equals(pq.read_table(b / "G4-1" / "train.parquet"))


def test_the_crawl_bound_drops_only_later_fineweb_rows(inputs):
    out = inputs["tmp"] / "out"
    g4arm.build(args_for(inputs, out, fineweb_max_crawl="2023"), counter=FakeCounter())
    report = json.loads((out / "report.json").read_text())
    assert report["bt"]["funnel"]["fineweb"]["crawl"] > 0
    assert all("crawl" not in c for s, c in report["bt"]["funnel"].items() if s != "fineweb")
    assert max(int(y) for y in report["set"]["bt_fineweb_crawl_years"]) <= 2023
    assert g4arm.crawl_key("CC-MAIN-2023-23") == (2023, 23) and g4arm.crawl_key("2023") == (2023, 99)
    assert g4arm.crawl_key("CC-MAIN-2023-40") > g4arm.crawl_key("CC-MAIN-2023-23")
    with pytest.raises(SystemExit):
        g4arm.crawl_key("2023-40")


def test_a_pinned_revision_that_differs_stops_the_build(inputs):
    revisions = json.loads((inputs["bt"] / "revisions.json").read_text())
    revisions["dpc"]["revision"] = "0" * 40
    (inputs["bt"] / "revisions.json").write_text(json.dumps(revisions))
    with pytest.raises(SystemExit, match="pins"):
        g4arm.build(args_for(inputs, inputs["tmp"] / "out"), counter=FakeCounter())


def test_replay_is_zero_by_default_and_never_translategemmas_flan_nl(inputs):
    tmp = inputs["tmp"]
    rows = [{"id": f"seg-{k}", "task": "t", "en": text(random.Random(k), EN_WORDS, 30),
             "nl": text(random.Random(k + 1), WORDS, 32), "model": "google/gemma-4-31B-it"} for k in range(200)]
    write(rows, tmp / "g4-pilot" / "replay.parquet")
    write([{"set": "classifier", "segments": ["seg-0", "seg-1"]}], tmp / "eval-rows.parquet")
    with pytest.raises(SystemExit, match="share is 0"):
        g4arm.build(args_for(inputs, tmp / "o1", replay=str(tmp / "g4-pilot" / "replay.parquet")), counter=FakeCounter())
    with pytest.raises(SystemExit, match="needs --replay"):
        g4arm.build(args_for(inputs, tmp / "o2", replay_share=0.1), counter=FakeCounter())
    for path, model in (("gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b/x.parquet", "google/gemma-4-31B-it"),
                        ("yhavinga/flan-nl", "google/gemma-4-31B-it"), ("/x/replay.parquet", "translategemma-27b")):
        with pytest.raises(SystemExit, match="§2.5"):
            g4arm.check_replay_source(path, model)
    g4arm.check_replay_source("gs://t5-dutch-english-usc2/mbert/flan-nl/g4-pilot/replay.parquet", "google/gemma-4-31B-it")
    out = tmp / "o3"
    g4arm.build(args_for(inputs, out, replay_share=0.1, replay=str(tmp / "g4-pilot" / "replay.parquet"),
                         replay_model="google/gemma-4-31B-it", flan_eval=str(tmp / "eval-rows.parquet")),
                counter=FakeCounter())
    metas = pq.read_table(out / "G4-1" / "train-meta.parquet").to_pylist()
    replay = [m for m in metas if m["kind"] == "replay"]
    assert len(replay) == round(600 * 0.1 / 0.9) == 67
    assert not {"replay:seg-0", "replay:seg-1"} & {m["id"] for m in replay}
    rows[5]["model"] = "google/translategemma-27b-it"
    write(rows, tmp / "g4-pilot" / "replay.parquet")
    with pytest.raises(SystemExit, match="not Gemma 4"):
        g4arm.build(args_for(inputs, tmp / "o4", replay_share=0.1, replay=str(tmp / "g4-pilot" / "replay.parquet"),
                             replay_model="google/gemma-4-31B-it", flan_eval=str(tmp / "eval-rows.parquet")),
                    counter=FakeCounter())


def test_a_ratio_the_bins_cannot_hold_stops_the_build(inputs, monkeypatch):
    monkeypatch.setattr(g4arm, "BT_PER_BOOK", 10)
    with pytest.raises(SystemExit, match="amendment"):
        g4arm.build(args_for(inputs, inputs["tmp"] / "out"), counter=FakeCounter())


def eval_inputs(tmp_path, monkeypatch, tokens):
    monkeypatch.setattr(g4arm, "FOUR_SETS", {"wmt24pp": 60, "flores": 4})
    monkeypatch.setattr(g4arm, "N_FLAN", 3)
    monkeypatch.setattr(g4arm, "N_BLIND", 2)
    monkeypatch.setattr(g4arm, "N_WMT_GOOD", 58)
    four = [{"id": f"wmt24pp-{k:04d}", "set": "wmt24pp", "source": f"s{k}", "reference": "r", "tokens": 1,
             "over_512": False, "length_class": "tiny", "meta": json.dumps({"is_bad_source": k < 2})} for k in range(60)]
    four += [{"id": f"flores-{k}", "set": "flores", "source": f"f{k}", "reference": "r", "tokens": 1, "over_512": False,
              "length_class": "tiny", "meta": "{}"} for k in range(4)]
    x = [{"id": f"flan:{k}", "set": "flan", "source": f"q{k}", "reference": None, "tokens": 1, "over_512": False,
          "length_class": "tiny", "meta": "{}"} for k in range(3)]
    x += [{"id": f"blind-{k:03d}", "set": "blind", "source": f"b{k}", "reference": None, "tokens": 1, "over_512": False,
           "length_class": "tiny", "meta": "{}"} for k in range(2)]
    x += [{"id": "indomain:1", "set": "indomain", "source": "i", "reference": "r", "tokens": 1, "over_512": False,
           "length_class": "tiny", "meta": "{}"}]
    write(four, tmp_path / "four.parquet")
    write(x, tmp_path / "x.parquet")
    args = type("Args", (), {"four": str(tmp_path / "four.parquet"), "x": str(tmp_path / "x.parquet"),
                             "tokenizer": "unused", "workers": 4, "out": str(tmp_path / "eval")})()

    def count(texts):
        return [tokens(t) for t in texts]

    return args, count


def test_evaldata_counts_in_gemma4_tokens_splits_whole_classes_and_picks_the_smoke_segments(tmp_path, monkeypatch):
    def tokens(t):  # WMT24++ small (the first 55) and medium, FLORES tiny, FLAN long, blind xlong: 5 classes
        if t.startswith("s"):
            return 100 if int(t[1:]) < 55 else 200
        return {"f": 20, "q": 300, "b": 450}[t[0]]

    args, count = eval_inputs(tmp_path, monkeypatch, tokens)
    g4arm.evaldata(args, count=count, check_prompts=lambda segments: {"natural-v1": 80})
    d = tmp_path / "eval"
    rows = pq.read_table(d / "evalsets.parquet").to_pylist()
    assert len(rows) == 69 and {r["set"] for r in rows} == {"wmt24pp", "flores", "flan", "blind"}
    assert {r["length_class"] for r in rows if r["set"] == "wmt24pp"} == {"small", "medium"}
    assert {r["length_class"] for r in rows if r["set"] == "blind"} == {"xlong"}
    assert {r["length_class"] for r in rows if r["set"] == "flan"} == {"long"}
    seen = []
    for k in range(4):
        seen += pq.read_table(d / f"segments-w{k}.parquet").column("id").to_pylist()
        classes = (d / f"classes-w{k}.txt").read_text().split()
        assert all(c in g4arm.EVAL_CLASSES for c in classes)
    assert sorted(seen) == sorted(r["id"] for r in rows)
    smoke = pq.read_table(d / "segments-smoke.parquet").column("id").to_pylist()
    assert smoke == [f"wmt24pp-{k:04d}" for k in range(2, 52)]


def test_evaldata_refuses_a_segment_above_class_xlong(tmp_path, monkeypatch):
    args, count = eval_inputs(tmp_path, monkeypatch, lambda t: 600 if t == "s7" else 100)
    with pytest.raises(SystemExit, match="xlong"):
        g4arm.evaldata(args, count=count, check_prompts=lambda segments: {})


def judgment(blind_id, fa, fb, verdict, doubt="zeker"):
    return {"blind_id": blind_id, "fout_A": fa, "fout_B": fb, "oordeel": verdict, "zekerheid": doubt, "uitleg": "."}


def test_tally_gives_a2_and_c_from_the_key():
    key = [{"blind_id": "b0", "A": "base", "B": "G4-1", "set": "wmt24pp"},
           {"blind_id": "b1", "A": "G4-1", "B": "base", "set": "idiom"},
           {"blind_id": "b2", "A": "base", "B": "G4-1", "set": "flan"},
           {"blind_id": "b3", "A": "G4-1", "B": "base", "set": "flan"},
           {"blind_id": "b4", "A": "base", "B": "G4-1", "set": "flan"}]
    judgments = [judgment("b0", "ja", "nee", "B beter"), judgment("b1", "nee", "nee", "A beter", "twijfel"),
                 judgment("b2", "nee", "ja", "A beter"), judgment("b3", "nee", "nee", "gelijk"),
                 judgment("b4", "nee", "nee", "B beter")]
    t = g4arm.tally_rows(judgments, key, "G4-1")
    assert (t["arm_better"], t["base_better"], t["equal"]) == (3, 1, 1)
    assert t["share_of_unequal"] == 0.75 and t["c"]
    assert (t["errors_arm"], t["errors_base"]) == (1, 1) and t["a2"] and t["doubt"] == 1
    judgments[0] = judgment("b0", "nee", "ja", "A beter")
    t = g4arm.tally_rows(judgments, key, "G4-1")
    assert (t["errors_arm"], t["errors_base"]) == (2, 0) and not t["a2"]
    assert t["share_of_unequal"] == 0.5 and not t["c"]
    with pytest.raises(SystemExit):
        g4arm.tally_rows(judgments[:4], key, "G4-1")


def test_rules_a1_and_b_on_their_bounds():
    sets = {s: [{"metricx_qe": v}] for s, v in g4arm.HUMAN_QE.items()}
    assert g4arm.rule_a1(sets)["holds"]
    sets["flores"] = [{"metricx_qe": g4arm.HUMAN_QE["flores"] + 0.001}]
    a1 = g4arm.rule_a1(sets)
    assert not a1["holds"] and a1["sets"]["wmt24pp"]["holds"] and not a1["sets"]["flores"]["holds"]
    shares = {"G4-1": {"pooled": 0.405}, "G4-0": {"pooled": 0.305}, "TG": {"pooled": 0.19}}
    assert g4arm.rule_b(shares)["holds"]
    shares["G4-1"]["pooled"] = 0.40
    assert not g4arm.rule_b(shares)["holds"] and g4arm.rule_b(shares, "G4-1", "TG")["holds"]


def smoke_rows(n, qe, ended=True):
    return [{"id": f"w{k}", "hypothesis": f"h{k}", "ended": ended, "decode_loop": False, "empty": False,
             "metricx_qe": qe, "metricx_ref": qe} for k in range(n)]


def test_the_smoke_verdict_needs_every_check():
    summary = {"steps_done": 8, "loss": [[k, 2.0] for k in range(1, 9)], "peak_gib_max": 24.3}
    adapter = {"rank": 32, "alpha": 16.0, "scale": 0.5}
    lora = {"modules": 410, "relative_update_norm": 1e-4, "elements_changed": 0.1}
    ok = g4arm.smoke_verdict(summary, adapter, True, lora, smoke_rows(50, 2.1), smoke_rows(50, 2.0))
    assert ok["passed"]
    for bad in ({"lora": {**lora, "modules": 0}}, {"checkpoint_ok": False}, {"arm_rows": smoke_rows(50, 3.0)},
                {"base_rows": smoke_rows(49, 2.0)}, {"summary": {**summary, "peak_gib_max": 29.6}},
                {"adapter": {**adapter, "rank": 64}}, {"arm_rows": smoke_rows(50, 2.0, ended=False)}):
        kwargs = {"summary": summary, "adapter": adapter, "checkpoint_ok": True, "lora": lora,
                  "arm_rows": smoke_rows(50, 2.1), "base_rows": smoke_rows(50, 2.0)}
        kwargs.update(bad)
        assert not g4arm.smoke_verdict(**kwargs)["passed"], bad


def test_lora_line_reads_translates_merge_line():
    log = 'x\nLORA {"dir": "/a", "rank": 32, "alpha": 16.0, "modules": 410, "relative_update_norm": 0.01}\nSHARD {}\n'
    assert g4arm.lora_line(log)["modules"] == 410 and g4arm.lora_line("nothing") is None


def test_report_reads_the_chains_outputs_and_decides_a1_and_b(tmp_path, monkeypatch):
    monkeypatch.setattr(g4arm, "N_WMT_GOOD", 2)
    monkeypatch.setattr(g4arm, "N_FLAN", 1)
    evalsets = [{"id": f"{s}-{k}", "set": s, "source": "a b c", "reference": None if s in ("flan", "blind") else "x y z",
                 "tokens": 3, "over_512": False, "length_class": "tiny", "meta": "{}"}
                for s in ("wmt24pp", "flores", "ntrex", "europarl", "flan", "blind") for k in range(2)]
    evalsets = [r for r in evalsets if r["id"] != "flan-1"]
    write(evalsets, tmp_path / "evalsets.parquet")
    for name, qe in (("G4-1", 1.0), ("G4-0", 2.0)):
        d = tmp_path / name
        pairs = [{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": "p q r", "reference": r["reference"],
                  "ended": True, "decode_loop": False, "english_left": False, "empty": False} for r in evalsets]
        write(pairs, d / "pairs" / "pairs-w0.parquet")
        write([{"id": r["id"], "qe_source_hypothesis": qe,
                "ref_source_hypothesis_reference": qe if r["reference"] else None} for r in evalsets],
              d / "scores" / "metricx-w0.parquet")
        write([{"id": r["id"], "comet22": 0.85} for r in evalsets if r["reference"]], d / "scores" / "comet-w0.parquet")
    write([{"id": r["id"], "set": r["set"], "metricx_qe": 1.5, "metricx_ref": 1.4, "comet22": 0.86}
           for r in evalsets if r["set"] in g4arm.FOUR_SETS], tmp_path / "tg.parquet")
    preds = []
    for model, p in (("G4-1", 0.9), ("G4-0", 0.1), ("TG", 0.1)):
        preds += [{"model": model, "set": s, "id": i, "p_original": p, "truncated": False}
                  for s, i in (("wmt24pp", "wmt24pp-0"), ("wmt24pp", "wmt24pp-1"), ("flan", "flan-0"))]
    write(preds, tmp_path / "pred.parquet")
    (tmp_path / "cls.json").write_text(json.dumps({"test_accuracy": 0.9575}))
    out = tmp_path / "report.md"
    args = type("Args", (), {"evalsets": str(tmp_path / "evalsets.parquet"), "arm": str(tmp_path / "G4-1"),
                             "base": str(tmp_path / "G4-0"), "tg_scores": str(tmp_path / "tg.parquet"),
                             "predictions": str(tmp_path / "pred.parquet"), "classifier_results": str(tmp_path / "cls.json"),
                             "lora_line": [], "train_summary": None, "tally": None, "out": str(out)})()
    g4arm.report(args)
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["a1"]["holds"] and not r["a1_base"]["sets"]["flores"]["holds"]
    assert r["b"]["holds"] and r["b"]["delta"] == pytest.approx(1.0) and r["classifier_accuracy"] == 0.9575
    assert r["verdict"].startswith("wacht op de blinde beoordeling")
    assert r["sets"]["wmt24pp"]["TG"]["metricx_qe"] == 1.5 and r["sets"]["flan"]["G4-1"]["n"] == 1
    assert "(a1) gehaald" in out.read_text() and "2,230" in out.read_text()
    tally = {"a2": True, "c": False, "arm": "G4-1", "arm_better": 40, "base_better": 30, "equal": 30,
             "share_of_unequal": 40 / 70, "errors_arm": 10, "errors_base": 12, "doubt": 3}
    (tmp_path / "tally.json").write_text(json.dumps(tally))
    args.tally = str(tmp_path / "tally.json")
    g4arm.report(args)
    assert json.loads(out.with_suffix(".json").read_text())["verdict"] == "helpt niet"
