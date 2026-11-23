"""The measurement C2F (protocol-g4.md §16): c2fiction.py's groups (whole in the encoder, the held-out books apart,
the anchor's novels paired with their -b1- rows), its AUC and its report, on synthetic inputs (no model, no TPU)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import c2fiction as cf  # noqa: E402

WORDS = lambda texts: [len(t.split()) for t in texts]  # noqa: E731
LONG = " ".join(["w"] * (cf.MAX_TOKENS + 1))
EXACT = " ".join(["w"] * cf.MAX_TOKENS)


def g41_pair(pid, nl, kind="book"):
    return ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": f" {nl} "}]},
            {"id": pid, "kind": kind})


def world():
    heldout = [{"id": f"heldout:books:book:books:1-2-b1-{k}", "set": "books", "reference": f"origineel een {k}"}
               for k in range(4)]
    heldout += [{"id": "heldout:books:book:books:3-4-b1-0", "set": "books", "reference": LONG},
                {"id": "heldout:btval:x", "set": "btval", "reference": "geen boek"}]
    g41 = [g41_pair(f"book:books:1-2-b1-{k}", f"origineel een {k}") for k in range(4)]
    g41 += [g41_pair("book:books:3-4-b1-0", LONG)]
    g41 += [g41_pair(f"book:books:5-6-b1-{k}", f"ander boek {k}") for k in range(3)]
    g41 += [g41_pair("book:books:7-8-b1-0", EXACT), g41_pair("bt:x", "bt", kind="bt")]
    anchor, b1 = [], []
    for k in range(5):
        anchor.append({"uid": f"novels:10-20-b1-{k}#0", "id": f"novels:10-20-b1-{k}", "j": 0, "source": f"en {k} ",
                       "hypothesis": f"g40 {k}", "ended": True, "decode_loop": False, "english_left": False,
                       "empty": False})
        b1.append({"id": f"10-20-b1-{k}", "group": "en_orig", "en": f"en {k}", "nl": f"mens {k}"})
    anchor[1]["ended"] = False                      # not eligible
    anchor[2]["hypothesis"] = LONG                  # G4-0's side too long
    b1[3]["nl"] = LONG                              # the human side too long
    anchor.append({**anchor[0], "uid": "flan:1#0", "id": "flan:1"})
    return heldout, g41, ["1-2", "3-4"], anchor, b1


def test_build_groups_pairs_and_whole_texts():
    heldout, g41, books, anchor, b1 = world()
    texts, meta, eligible, report = cf.build(heldout, g41, books, anchor, b1, [WORDS, WORDS])
    by = {}
    for t in texts:
        by.setdefault(t["model"], []).append(t)
    assert sorted(by) == ["g40", "human", "nl_books", "nl_books31"]
    assert {t["set"] for t in texts} == {"fiction"}
    # nl_books: the 4 fitting rows of set books (the long one out); nl_books31 without the held-out books 1-2 and 3-4
    assert [t["id"] for t in by["nl_books"]] == [f"heldout:books:book:books:1-2-b1-{k}" for k in range(4)]
    assert report["groups"]["nl_books"]["too_long"] == 1
    assert {m["book"] for m in meta if m["group"] == "nl_books31"} == {"5-6", "7-8"}
    assert report["groups"]["nl_books31"]["eligible"] == 4       # EXACT fits (1,022 tokens)
    assert report["nl_books_equal_to_g41"] == 4
    # human/g40: paired, the not-ended row and both too-long pairs out
    assert [t["id"] for t in by["human"]] == [t["id"] for t in by["g40"]] == ["novels:10-20-b1-0", "novels:10-20-b1-4"]
    assert [t["text"] for t in by["human"]] == ["mens 0", "mens 4"]
    assert [t["text"] for t in by["g40"]] == ["g40 0", "g40 4"]
    nov = report["groups"]["novels"]
    assert (nov["anchor_novels"], nov["anchor_eligible"], nov["human_too_long"], nov["g40_too_long"]) == (5, 4, 1, 1)
    assert eligible == ["novels:10-20-b1-0", "novels:10-20-b1-2", "novels:10-20-b1-3", "novels:10-20-b1-4"]
    assert all("text" not in m for m in meta)
    # the draw is seeded: the same inputs give the same texts
    assert cf.build(*world()[:4], world()[4], [WORDS, WORDS])[0] == texts


def test_build_stops_on_a_different_english_or_a_missing_row():
    heldout, g41, books, anchor, b1 = world()
    b1[0]["en"] = "other"
    with pytest.raises(SystemExit, match="English"):
        cf.build(heldout, g41, books, anchor, b1, [WORDS, WORDS])
    heldout, g41, books, anchor, b1 = world()
    with pytest.raises(SystemExit, match="rows in b1-selected"):
        cf.build(heldout, g41, books, anchor, b1[1:], [WORDS, WORDS])


def test_draw_takes_at_most_n_and_is_seeded():
    rows = [{"id": f"{k:04d}"} for k in range(1200)]
    a, b = cf.draw(rows, cf.N, "s"), cf.draw(list(reversed(rows)), cf.N, "s")
    assert len(a) == cf.N and a == b and a != cf.draw(rows, cf.N, "t")
    assert len(cf.draw(rows[:7], cf.N, "s")) == 7


def test_auc():
    assert cf.auc([1, 2], [0, 0]) == 1.0
    assert cf.auc([0, 0], [1, 2]) == 0.0
    assert cf.auc([0.5], [0.5]) == 0.5
    assert cf.auc([1.0, 0.0], [0.5]) == 0.5
    assert cf.auc([0.9, 0.8, 0.3], [0.8, 0.1]) == pytest.approx((2 + 1.5 + 1) / 6)


def synthetic(n=40):
    meta, preds = [], {"C1": [], "C2": []}
    p = {"nl_books": 0.9, "nl_books31": 0.8, "human": 0.6, "g40": 0.1}
    for g in cf.GROUPS:
        for k in range(n):
            i = f"novels:{k}" if g in cf.PAIRED else f"{g}:{k}"
            meta.append({"group": g, "id": i, "book": f"b{k % 3}", "tokens_c1": 100 + 5 * k, "tokens_c2": 100 + 5 * k})
            for c in cf.CLASSIFIERS:
                x = p[g] if c == "C1" else 1 - p[g]
                preds[c].append({"model": g, "set": "fiction", "id": i, "p_original": x + (k % 5) * 0.01,
                                 "truncated": False})
    judge = [{"uid": f"novels:{k}#0", "p_c2": 0.9 + (k % 5) * 0.01, "logit_c2": 2.0, "c2_tokens": 100 + 5 * k,
              "c2_overflow": False} for k in range(n)]
    return meta, preds, judge


def test_measures_and_report():
    meta, preds, judge = synthetic()
    r = cf.measures(meta, preds, judge, [f"novels:{k}" for k in range(50)])
    assert r["groups"]["nl_books"]["C1"]["share_original"] == 1.0
    assert r["groups"]["g40"]["C1"]["share_original"] == 0.0
    assert r["groups"]["g40"]["C2"]["share_original"] == 1.0
    assert r["auc"]["human-g40"]["C1"]["auc"] == 1.0 and r["auc"]["human-g40"]["C1"]["paired"]
    assert r["auc"]["human-g40"]["C2"]["auc"] == 0.0
    assert r["auc"]["nl_books-human"]["C1"]["ci95"] == [1.0, 1.0]
    assert r["paired"]["C1"]["share_human_higher"] == 1.0 and r["paired"]["C1"]["mean_dlogit"] > 0
    assert r["paired"]["C1"]["ci95"][0] <= r["paired"]["C1"]["mean_dlogit"] <= r["paired"]["C1"]["ci95"][1]
    ch = r["checks"]["c2_vs_judge_on_g40"]
    assert (ch["compared"], ch["missing"], ch["over_tol"], ch["tokens_equal"]) == (40, 0, 0, 40)
    assert ch["max_abs_dp"] == pytest.approx(0.0, abs=1e-12)
    assert r["anchor_novels_judge_c2"]["scored"] == 40 and r["anchor_novels_judge_c2"]["share_original"] == 1.0
    assert sum(r["bands"][b]["nl_books"]["n"] for b in r["bands"]) == 40
    info = {"nl_books_equal_to_g41": 40, "groups": {"nl_books": {"rows": 41, "too_long": 1, "eligible": 40,
                                                                 "books_eligible": 3, "drawn": 40}}}
    md = cf.report_md(r, info)
    assert "| nl_books | 40 | 3 |" in md and "human-g40 (gepaard)" in md
    # the same measures again: the bootstrap is seeded
    assert cf.measures(meta, preds, judge, [f"novels:{k}" for k in range(50)]) == r


def test_measures_stops_on_a_truncated_short_text_and_on_missing_predictions():
    meta, preds, judge = synthetic()
    preds["C2"][0]["truncated"] = True
    with pytest.raises(SystemExit, match="truncated"):
        cf.measures(meta, preds, judge, [])
    meta, preds, judge = synthetic()
    meta[0]["tokens_c2"] = cf.MAX_TOKENS
    preds["C2"][0]["truncated"] = True              # exactly 1,022 tokens: flagged by naturalness.py, not cut
    assert cf.measures(meta, preds, judge, [])["checks"]["truncated_flags"]["C2"] == 1
    meta, preds, judge = synthetic()
    with pytest.raises(SystemExit, match="not the texts"):
        cf.measures(meta, {"C1": preds["C1"][1:], "C2": preds["C2"]}, judge, [])
