"""G4-maat2 (protocol-g4.md §24, part 1): g4measure2.py (the prompts A1 and J-nat as §24 has them, the texts parsed from
§20's prompts and checked against §20's and FB.8's, §23's prompts checked, the combinations A1-veto -> N1 / N2, the
flips, the reproduction and the test of G4-maat2.5) on small synthetic inputs, and run_g4measure2.sh's stages rendered
with a fake bucket (no model, no TPU)."""

import json
import os
import random
import re
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4measure as gm  # noqa: E402
import g4measure2 as m2  # noqa: E402
import pairjudge  # noqa: E402
from test_en_nl_g4arm8t import FAKE_GSUTIL  # noqa: E402
from test_en_nl_g4measure import answers, build_world, data_args, fake_scores  # noqa: E402
from test_en_nl_g4rl import FakeCounter7  # noqa: E402

WORDS = ("de", "het", "huis", "rivier", "groen", "loopt", "langzaam", "en", "stad", "oud", "licht", "water", "ziet")


def sentence(rng, n=10):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + f" w{rng.randrange(10 ** 9)}"


# ---------------------------------------------------------------------------------------------------- the protocol

def code_blocks(text):
    return re.findall(r"```\n(.*?)\n```", text, flags=re.S)


def test_section_24s_prompts_are_the_codes_letter_for_letter():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 24. Meting G4-maat2"):]
    blocks = code_blocks(section)
    assert blocks[0] == m2.A1_PROMPT and blocks[1] == m2.JNAT_PROMPT
    assert "≥ **0,2**" in section and m2.VETO == 0.2
    assert "> **70%**" in section and "≥ **65%**" in section and "≤ **20%**" in section
    assert (m2.TEST_A, m2.TEST_B, m2.TEST_C) == (0.70, 0.65, 0.20) and m2.CLAUDE_FAULT_PAIRS == 263
    assert '"en-nl-g4arm/0/G4-maat2/repro"' in section and m2.REPRO_SEED == "en-nl-g4arm/0/G4-maat2/repro"
    for prompt in (m2.A1_PROMPT, m2.JNAT_PROMPT):
        assert prompt.count("{bron}") == prompt.count("{A}") == prompt.count("{B}") == 1
        assert prompt.endswith("Antwoord met één woord: A, B of gelijk.")


# ---------------------------------------------------------------------------------------------------- parsing

def test_parse_prompt_gives_the_three_texts_back_and_stops_when_it_cannot():
    src, a, b = "An English source.", "Tekst A met\n\nalinea's.", "Tekst B."
    text = pairjudge.user_text(src, a, b)
    assert m2.parse_prompt(text) == (src, a, b)
    # the N1 template's parts
    assert m2.parse_prompt(pairjudge.user_text(src, a, b, pairjudge.N1_PROMPT), pairjudge.N1_PROMPT) == (src, a, b)
    with pytest.raises(SystemExit, match="exactly once"):
        m2.parse_prompt(pairjudge.user_text(src, a + "\n\nVertaling A:\nnog", b))
    with pytest.raises(SystemExit, match="not a prompt"):
        m2.parse_prompt(text[1:])
    with pytest.raises(SystemExit, match="not a prompt"):
        m2.parse_prompt(pairjudge.user_text(src, a, b, pairjudge.N1_PROMPT))


# ---------------------------------------------------------------------------------------------------- choices

def p(a, b):
    return {"A": a, "B": b, "gelijk": 1 - a - b}


def test_the_combination_lets_a1_decide_only_at_a_clear_margin():
    assert m2.combo_side(p(0.7, 0.2), p(0.1, 0.8)) == "A"            # |0.5| >= 0.2: A1
    assert m2.combo_side(p(0.3, 0.5), p(0.6, 0.3)) == "B"            # exactly 0.2: A1
    assert m2.combo_side(p(0.45, 0.4), p(0.1, 0.8)) == "B"           # 0.05: N1
    assert m2.combo_side(p(0.45, 0.4), p(0.4, 0.4)) is None          # N1 tied
    assert m2.combo_side(None, p(0.1, 0.8)) is None and m2.combo_side(p(0.9, 0.0), None) is None


def score_row(pa_, pb_, peq=None, skipped=None):
    if skipped:
        return {"skipped": skipped, "p_a": None, "p_b": None, "p_eq": None}
    return {"skipped": None, "p_a": pa_, "p_b": pb_, "p_eq": 1 - pa_ - pb_ if peq is None else peq}


def test_flips_of_a_prompt_and_of_a_combination():
    # a prompt: order ab says A, order ba says A (= B in the pair's frame): a flip
    assert m2.flip_of_prompt(score_row(0.7, 0.2), score_row(0.7, 0.2)) is True
    assert m2.flip_of_prompt(score_row(0.7, 0.2), score_row(0.2, 0.7)) is False
    assert m2.flip_of_prompt(score_row(0.1, 0.2), score_row(0.7, 0.2)) is False    # gelijk against a winner
    assert m2.flip_of_prompt(score_row(0, 0, skipped="too_long"), score_row(0.7, 0.2)) is None
    # a combination: A1 clear in both orders and consistent: no flip; A1 unclear, N1 position A in both: a flip
    a1 = (score_row(0.8, 0.1), score_row(0.1, 0.8))
    n1 = (score_row(0.6, 0.3), score_row(0.6, 0.3))
    assert m2.flip_of_combo(*a1, *n1) is False
    weak = (score_row(0.45, 0.4), score_row(0.45, 0.4))
    assert m2.flip_of_combo(*weak, *n1) is True
    assert m2.flip_of_combo(*weak, score_row(0, 0, skipped="x"), n1[1]) is None


# ---------------------------------------------------------------------------------------------------- data1

def fb_world(tmp, rng):
    """§20's refs and prompts of 1,000 pairs and FB.8's N1/N2 prompts of the same pairs"""
    refs, rubric, n1, n2 = [], [], [], []
    for k in range(m2.FB_PAIRS):
        test = "fiction" if k < 500 else "bt"
        i = f"novels:{k}" if test == "fiction" else f"dpc:doc{k}:0"
        refs.append({"package": test, "blind_id": i, "stratum": "s", "tokens_en": 30, "identical": False})
        src, a, b = sentence(rng), sentence(rng), sentence(rng)
        key = f"{test}/{i}"
        for out, template in ((rubric, pairjudge.PROMPT), (n1, pairjudge.N1_PROMPT), (n2, pairjudge.N2_PROMPT)):
            out += [{"uid": u, "text": t} for u, t, _ in pairjudge.pair_prompts(key, src, a, b, template)]
    d = tmp / "fb"
    pairjudge.write_rows(refs, d / "refs.parquet")
    for name, rows in (("fb", rubric), ("fbn1", n1), ("fbn2", n2)):
        pairjudge.write_prompts(rows, d / name)
    return d


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m2")
    blind, inp = build_world(tmp)
    out23 = tmp / "out23"
    gm.data(data_args(blind, inp, out23), counter=FakeCounter7())
    fb = fb_world(tmp, random.Random(5))
    return tmp, blind, out23, fb


def data1_args(fb, out23, out):
    return Namespace(fb_refs=str(fb / "refs.parquet"), fb_prompts=str(fb / "fb" / "prompts-w*.parquet"),
                     fb_n1_prompts=str(fb / "fbn1" / "prompts-w*.parquet"), fb_n2_prompts=str(fb / "fbn2" / "prompts-w*.parquet"),
                     pairs=str(out23 / "pairs.parquet"), m23_prompts=str(out23 / "prompts-w*.parquet"), out=str(out))


def test_data1_builds_a1_and_jnat_of_both_sets_and_the_reproduction(world, tmp_path, capsys):
    tmp, blind, out23, fb = world
    info = m2.data1(data1_args(fb, out23, tmp_path / "out"))
    assert (info["fb_prompts"], info["m23_prompts"], info["repro_prompts"], info["prompts"]) == (4000, 5752, 1200, 10952)
    rows = [r for k in range(4) for r in pq.read_table(tmp_path / "out" / f"prompts-w{k}.parquet").to_pylist()]
    assert len(rows) == 10952 and len({r["uid"] for r in rows}) == 10952
    for k in range(4):
        assert all(pairjudge.worker_of(r["uid"]) == k for r in pq.read_table(tmp_path / "out" / f"prompts-w{k}.parquet").to_pylist())
    by = {r["uid"]: r["text"] for r in rows}
    # A1 of a fiction pair: the parsed texts in the new template; J-nat of a §23 pair
    a1 = by["fb/a1/fiction/novels:0/ab"]
    assert a1.startswith("Je beoordeelt vertalingen") and "Oordeel alleen op trouw aan de bron" in a1
    src, a, b = m2.parse_prompt(a1, m2.A1_PROMPT)
    assert by["fb/a1/fiction/novels:0/ba"] == pairjudge.user_text(src, b, a, m2.A1_PROMPT)
    assert by["G4-7/blind-001/jnat/ab"].endswith("Welke vertaling kies je? Antwoord met één woord: A, B of gelijk.")
    # the reproduction: §23's own texts
    m23 = {r["uid"]: r["text"] for k in range(4) for r in pq.read_table(out23 / f"prompts-w{k}.parquet").to_pylist()}
    repro = [u for u in by if u.startswith("repro/")]
    assert len(repro) == 1200 and all(by[u] == m23[u[len("repro/"):]] for u in repro)
    assert len(info["repro_pairs"]) == 200 and "MEASURE2DATA" in capsys.readouterr().out


def test_data1_stops_on_a_prompt_that_is_not_20s_fb8s_or_23s(world, tmp_path):
    tmp, blind, out23, fb = world
    bad = tmp_path / "fb"
    for name in ("fb", "fbn1", "fbn2"):
        (bad / name).mkdir(parents=True)
        for k in range(4):
            (bad / name / f"prompts-w{k}.parquet").write_bytes((fb / name / f"prompts-w{k}.parquet").read_bytes())
    (bad / "refs.parquet").write_bytes((fb / "refs.parquet").read_bytes())
    f = bad / "fbn1" / "prompts-w0.parquet"
    rows = pq.read_table(f).to_pylist()
    rows[0]["text"] = rows[0]["text"].replace("Tekst B:", "Tekst  B:")
    pairjudge.write_rows(rows, f)
    with pytest.raises(SystemExit, match="n1 prompt of §20/FB.8"):
        m2.data1(data1_args(bad, out23, tmp_path / "out"))
    assert not (tmp_path / "out").exists()
    bad23 = tmp_path / "o23"
    bad23.mkdir()
    (bad23 / "pairs.parquet").write_bytes((out23 / "pairs.parquet").read_bytes())
    for k in range(4):
        rows = pq.read_table(out23 / f"prompts-w{k}.parquet").to_pylist()
        if k == 1:
            rows[3]["text"] += " "
        pairjudge.write_rows(rows, bad23 / f"prompts-w{k}.parquet")
    with pytest.raises(SystemExit, match="not §23's prompt"):
        m2.data1(data1_args(fb, bad23, tmp_path / "out"))


# ---------------------------------------------------------------------------------------------------- report1

def new_scores(rows_by_uid, fb_side, m23_refs, judge23):
    """Scores of the new prompts: A1 picks the human side on fiction/BT clearly, J-nat a weak position-A answer; on §23
    A1 copies the rubric of §23's fake scores, J-nat position A; the reproduction §23's own scores"""
    out = []
    for uid in rows_by_uid:
        key, order = uid.rsplit("/", 1)
        if uid.startswith("repro/"):
            out.append({**judge23[uid[len("repro/"):]], "uid": uid})
            continue
        if uid.startswith("fb/"):
            name = uid.split("/")[1]
            if name == "a1":
                pa_, pb_ = (0.8, 0.1) if order == "ab" else (0.1, 0.8)
            else:
                pa_, pb_ = 0.45, 0.4
            out.append({"uid": uid, "skipped": None, "p_a": pa_, "p_b": pb_, "p_eq": 1 - pa_ - pb_})
            continue
        pair, name = key.rsplit("/", 1)
        if name == "a1":
            r = judge23[f"{pair}/rubric/{order}"]
            out.append({**r, "uid": uid})
        else:
            pa_, pb_ = (0.6, 0.2) if order == "ab" else (0.2, 0.6)
            out.append({"uid": uid, "skipped": None, "p_a": pa_, "p_b": pb_, "p_eq": 0.2})
    # one schema for every row, as pairjudge.py score writes them
    return [{**r, "mass": 0.9, "top_ids": [1]} for r in out]


def fb_old_scores(fb_dir, tmp):
    """§20's rubric (the machine side), FB.8's N1 (the human side) and N2 (position A) on the 1,000 pairs"""
    refs = pq.read_table(fb_dir / "refs.parquet").to_pylist()
    out = {}
    for name in ("rubric", "n1", "n2"):
        rows = []
        for r in refs:
            for order in ("ab", "ba"):
                if name == "rubric":
                    pa_, pb_ = (0.2, 0.7) if order == "ab" else (0.7, 0.2)
                elif name == "n1":
                    pa_, pb_ = (0.7, 0.2) if order == "ab" else (0.2, 0.7)
                else:
                    pa_, pb_ = 0.6, 0.3
                rows.append({"uid": f"{r['package']}/{r['blind_id']}/{order}", "skipped": None, "p_a": pa_, "p_b": pb_,
                             "p_eq": 1 - pa_ - pb_, "mass": 0.9, "top_ids": [1]})
        pairjudge.write_rows(rows, tmp / f"fb-{name}" / "shard-00000.parquet")
        out[name] = str(tmp / f"fb-{name}" / "shard-*.parquet")
    return out


def report1_args(tmp, out23, d, blind, fb, new_glob, data_json, fb_globs):
    return Namespace(refs=str(out23 / "refs.parquet"), judge=str(d / "judge" / "shard-*.parquet"),
                     metricx=str(d / "metricx.parquet"), comet=str(d / "comet.parquet"), c1=str(d / "c1.parquet"),
                     c2=str(d / "c2.parquet"), new=new_glob, fb_refs=str(fb / "refs.parquet"),
                     fb_scores=[f"{k}={v}" for k, v in fb_globs.items()], data_json=str(data_json),
                     yeb1_key=str(blind / gm.YEB1 / "key-map.json"), yeb1_answers=str(tmp / "a1.txt"),
                     yeb2_key=str(blind / gm.YEB2 / "key.json"), yeb2_answers=str(tmp / "a2.txt"),
                     out=str(tmp / "report" / "report.md"))


def report_world(world, tmp_path, monkeypatch):
    tmp, blind, out23, fb = world
    info = m2.data1(data1_args(fb, out23, tmp_path / "data1"))
    refs = pq.read_table(out23 / "refs.parquet").to_pylist()
    side = answers(tmp_path, blind)
    d = fake_scores(out23, refs, side, tmp_path)
    judge23 = pairjudge.scores_by_uid(pairjudge.read_glob(str(d / "judge" / "shard-*.parquet")))
    prompts = {r["uid"]: r["text"] for k in range(4)
               for r in pq.read_table(tmp_path / "data1" / f"prompts-w{k}.parquet").to_pylist()}
    pairjudge.write_rows(new_scores(prompts, None, refs, judge23), tmp_path / "new" / "shard-00000.parquet")
    # §23's fake world has 1,438 pairs but its own count of one-sided Claude errors
    fault = sum(r["origin"] == "package" and r["claude_fout_a"] != r["claude_fout_b"] for r in refs)
    monkeypatch.setattr(m2, "CLAUDE_FAULT_PAIRS", fault)
    args = report1_args(tmp_path, out23, d, blind, fb, str(tmp_path / "new" / "shard-*.parquet"),
                        tmp_path / "data1" / "data.json", fb_old_scores(fb, tmp_path))
    return args, info, judge23, prompts


def test_report1_tests_a1_jnat_and_the_combinations(world, tmp_path, monkeypatch, capsys):
    args, info, _, _ = report_world(world, tmp_path, monkeypatch)
    r = m2.report1(args)
    t = r["test"]
    # A1 picks the human side clearly in both orders: (a) 100%, no flips; J-nat says A weakly in both orders: a flip
    # where its greedy answers differ (here both A in their own order: every pair flips), direction 100% (P̄ equal = ½?)
    assert t["A1"]["a"]["share"] == 1.0 and t["A1"]["a_pass"] and t["A1"]["c_fb"]["share"] == 0.0
    assert t["J-nat"]["c_fb"]["share"] == 1.0 and not t["J-nat"]["c_pass"]
    assert t["J-nat"]["a"]["share"] == 0.5 and not t["J-nat"]["a_pass"]
    # A1 decides everywhere on fiction/BT (margin 0.7): the combinations follow A1
    assert t["A1-veto → N1"]["a"]["share"] == 1.0 and t["A1-veto → N2"]["a"]["share"] == 1.0
    # the old prompts from §20's and FB.8's scores: the rubric the machine, N1 the human side
    assert t["rubriek"]["a"]["share"] == 0.0 and t["N1"]["a"]["share"] == 1.0
    assert r["fb"]["A1"]["fiction"]["n"] == 500 and r["fb"]["A1"]["bt"]["direction"]["n"] == 500
    # on §23: A1 is the rubric of §23's fake scores, so the same agreement in every column
    m = r["tables"]["main"]
    for col in m:
        assert m[col]["A1"]["k"] == m[col]["rubriek"]["k"] and m[col]["A1"]["n"] == m[col]["rubriek"]["n"]
    assert m["Yeb samen"]["J-nat"]["k"] == m["Yeb samen"]["positie A"]["k"]
    assert "A1-veto → N1" in r["tables"]["faults"]["Claude (fout aan één kant)"]
    assert t["A1"]["b"]["n_ref"] == m2.CLAUDE_FAULT_PAIRS
    assert r["checks"]["repro"]["deviations"] == 0 and r["checks"]["repro"]["max_abs"] == 0
    md = (tmp_path / "report" / "report.md").read_text()
    assert md.startswith("# G4-maat2") and "| A1-veto → N1 |" in md and "## 1. Vooraf vastgelegde toets" in md
    pairs = pq.read_table(world[2] / "pairs.parquet").to_pylist()
    words = {w for x in pairs[:100] for w in x["A"].split() if w.startswith("w")}
    assert not any(w in md for w in words)
    assert "MEASURE2REPORT" in capsys.readouterr().out


def test_report1_stops_when_the_reproduction_is_not_23s(world, tmp_path, monkeypatch):
    args, info, judge23, prompts = report_world(world, tmp_path, monkeypatch)
    rows = pq.read_table(tmp_path / "new" / "shard-00000.parquet").to_pylist()
    for row in rows:
        if row["uid"].startswith("repro/") and row["uid"].endswith("/ab") and not row["skipped"]:
            row["p_a"] += 2e-5
            row["p_b"] -= 2e-5
            break
    pairjudge.write_rows(rows, tmp_path / "new" / "shard-00000.parquet")
    with pytest.raises(SystemExit, match="reproduction deviates"):
        m2.report1(args)
    # within float32 precision it passes
    for row in rows:
        if row["uid"].startswith("repro/") and row["uid"].endswith("/ab") and not row["skipped"]:
            row["p_a"] -= 2e-5 - 5e-7
            row["p_b"] += 2e-5 - 5e-7
            break
    pairjudge.write_rows(rows, tmp_path / "new" / "shard-00000.parquet")
    r = m2.report1(args)
    assert 0 < r["checks"]["repro"]["max_abs"] <= m2.REPRO_TOL


# ---------------------------------------------------------------------------------------------------- run_g4measure2.sh

def stage_world(tmp_path):
    shm, gs, home, fake, code = (tmp_path / x for x in ("shm", "gs", "home", "bin", "code"))
    for d in (code, home, fake):
        d.mkdir(parents=True)
    for name in ("run_g4measure2.sh", "run_arms.sh"):
        (code / name).write_text((CODE / name).read_text().replace("/dev/shm/", f"{shm}/"))
    for f in CODE.glob("*.py"):
        (code / f.name).symlink_to(f)
    for name, body in {"gsutil": FAKE_GSUTIL, "sudo": "#!/bin/bash\nexit 0\n", "hostname": "#!/bin/bash\necho t1v-n-x-w-0\n",
                       "setsid": '#!/bin/bash\necho "SETSID $*" >> "$FAKELOG"\n'}.items():
        (fake / name).write_text(body)
        (fake / name).chmod(0o755)
    for py in ("translate-generic/venv-fork-f4d8615f4/bin/python", "translator/smoke-lora/venv/bin/python",
               "translator/comet-venv/bin/python"):
        pth = shm / py
        pth.parent.mkdir(parents=True)
        pth.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        pth.chmod(0o755)
    for tok in ("translate-generic/tokenizer-gemma4", "translator/g4lora/tok-gemma4-31b"):
        (shm / tok).mkdir(parents=True)
        (shm / tok / "tokenizer.json").write_text("{}")
    (shm / "translator/metricx/metricx-24-hybrid-xxl-v2p6-bfloat16").mkdir(parents=True)
    g = gs / "t5-dutch-english-usc2/mbert/translator/sft/g4arm"
    calls = tmp_path / "calls.log"

    def put(rel, text="x"):
        f = g / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)

    def run(stage, host="0", **env):
        (fake / "hostname").write_text(f"#!/bin/bash\necho t1v-n-x-w-{host}\n")
        e = {"HOME": str(home), "PATH": f"{fake}:{os.environ['PATH']}", "FAKEGS": str(gs), "FAKELOG": str(calls),
             "TAG": "t1", **env}
        calls.write_text("")
        out = subprocess.run(["bash", str(code / "run_g4measure2.sh"), stage], capture_output=True, text=True,
                             timeout=120, env=e, cwd=home)
        deadline = time.time() + 30
        while "started " in out.stdout and "SETSID" not in calls.read_text() and time.time() < deadline:
            time.sleep(0.2)
        return out

    def body(name):
        f = home / f"job_body_{name}.sh"
        check = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
        return f.read_text()

    return run, body, put


def test_run_g4measure2_renders_its_jobs(tmp_path):
    assert subprocess.run(["bash", "-n", str(CODE / "run_g4measure2.sh")]).returncode == 0
    run, body, put = stage_world(tmp_path)
    out = run("data1", host="1")
    assert out.returncode != 0 and "data1 runs on w0 only" in out.stdout
    put("judge/fb/data/refs.parquet")
    put("measure/data/pairs.parquet")
    out = run("data1", TAG="d1")
    assert out.returncode == 0 and "started g4m2_data1_d1 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4m2_data1_d1")
    for x in ("g4measure2.py data1", "/judge/fb/data/prompts-w*.parquet", "/judge/fbn1/data/prompts-w*.parquet",
              "/judge/fbn2/data/prompts-w*.parquet", "/measure/data/prompts-w*.parquet", "/measure/data/pairs.parquet",
              "/g4arm/measure2/data/", "MEASURE2DATA_DONE"):
        assert x in b, x
    assert "rm " not in b
    # judge: the three sets, pairjudge.py score unchanged
    out = run("judge", host="2", TAG="j0", SET="x")
    assert out.returncode != 0 and "SET is m2, p1 or p2" in out.stdout
    out = run("judge", host="2", TAG="j1", SET="m2")
    assert out.returncode != 0 and "no gs://" in out.stdout
    put("measure2/data/prompts-w2.parquet")
    put("pool/judge2/data/prompts-w2.parquet")
    out = run("judge", host="2", TAG="j2", SET="m2")
    b = body("g4m2_judge_m2_j2")
    assert out.returncode == 0 and "pairjudge.py score --prompts" in b and "/measure2/judge/scores/w2" in b
    out = run("judge", host="2", TAG="j3", SET="p2")
    assert "/pool/judge2/scores/w2" in body("g4m2_judge_p2_j3")
    # the pool's data: G4-8.15's inputs, the evaluation, Yeb's keys
    out = run("pooldata", TAG="p1")
    b = body("g4p_data_p1")
    for x in ("g4pool.py data", "eval-novels-meta.parquet", "b1-selected.parquet", "/rl/G4-7/data/pool/pool.parquet",
              "/data/G4-8/fresh/fresh.parquet", "/data/G4-8/targets/targets.parquet", "/input/bt3", "split.json",
              "/mixsample/data/data.json", "/steekproef-yeb-g4-8-mix/key.json", "/steekproef-yeb/key-map.json",
              "/G4-judge-anchor-i15/key.jsonl", "/steekproef-yeb-g4-8/key-map.json", "/g4arm/pool/data/"):
        assert x in b, x
    # generate: samples on w0 only with the worker's seed; greedy with the LoRA checked
    out = run("generate", host="1", TAG="g0", JOB="samples")
    assert out.returncode != 0 and "generate runs on w0 only" in out.stdout
    put("pool/data/segments.parquet")
    out = run("generate", TAG="g1", JOB="samples")
    b = body("g4p_gen_samples_G4-0_g1")
    assert out.returncode == 0 and "--copies 8 --sample samples=1.0:" in b and "--length-class medium" in b
    assert "/g4arm/pool/{sample}/w0" in b and "--lora" not in b
    out = run("generate", host="2", TAG="g2", JOB="greedy", MODEL="G4-9")
    assert out.returncode != 0 and "MODEL is one of" in out.stdout
    put("G4-8-s140/adapter.npz")
    put("G4-8-s140/adapter.json", json.dumps({"rank": 32, "alpha": 16.0, "scale": 0.5}))
    out = run("generate", host="2", TAG="g3", JOB="greedy", MODEL="G4-8-s140")
    b = body("g4p_gen_greedy_G4-8-s140_g3")
    assert out.returncode == 0 and "--copies 1" in b and "--lora" in b and "/pool/gen/G4-8-s140/w2" in b and "rm " not in b
    put("G4-7-i15/adapter.npz")
    put("G4-7-i15/adapter.json", json.dumps({"rank": 16, "alpha": 16.0, "scale": 1.0}))
    out = run("generate", host="3", TAG="g4", JOB="greedy", MODEL="G4-7-i15")
    assert out.returncode != 0 and "not rank 32" in out.stdout
    # texts, qe, cands, choose, poolreport
    for f in ("pool/samples/w0/DONE", "pool/gen/G4-0/w1/DONE", "pool/gen/G4-8-s140/w2/DONE", "pool/gen/G4-7-i15/w3/DONE",
              "measure/data/qe-in.parquet"):
        put(f)
    out = run("texts", TAG="t2")
    b = body("g4p_texts_t2")
    assert out.returncode == 0 and "g4pool.py texts" in b and "--greedy G4-7-i15=" in b and "/pool/naturalness/texts.parquet" in b
    put("pool/texts/qe-in-w3.parquet")
    out = run("qe", host="3", TAG="q1")
    b = body("g4p_qe_q1")
    assert out.returncode == 0 and "metricx_qe.py score" in b and "--empty-candidates" not in b and "/pool/scores/qe-w3" in b
    for k in range(4):
        put(f"pool/scores/qe-w{k}.parquet")
    put("pool/texts/texts.parquet")
    out = run("cands", TAG="c1")
    b = body("g4p_cands_c1")
    assert out.returncode == 0 and "g4pool.py cands" in b and "/pool/judge1/data/" in b
    put("pool/choose/cands.parquet")
    out = run("choose", TAG="c2")
    assert "g4pool.py choose" in body("g4p_choose_c2") and "/pool/judge2/data/" in body("g4p_choose_c2")
    for f in ("pool/choose/choose.json", "pool/scores/comet.parquet", "pool/naturalness/predictions.parquet",
              "pool/naturalness/predictions-c2.parquet"):
        put(f)
    out = run("poolreport", TAG="r1")
    b = body("g4p_report_r1")
    assert out.returncode == 0 and "g4pool.py report" in b and "/measure/scores/metricx.parquet" in b
    # only files without text reach the report
    for x in ("texts.parquet ", "outputs.parquet ", "/pairs.parquet ", "sources.parquet "):
        assert x not in b, x
    put("pool/texts/comet-in.parquet")
    out = run("comet", host="2", TAG="m1")
    assert out.returncode == 0 and "/pool/scores/comet.parquet" in body("g4p_comet_m1")
