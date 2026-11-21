"""Arm G4-7's judges (protocol-g4.md §14, G4-7.4a): g4judge.py's stores, the sample and score files, the MetricX split,
C2's token rule (never truncated, more than 1,022 tokens is c2_overflow), streaming and idempotent scoring, the DONE
markers with the samples' sha256, the heartbeat, and the chain's checks (alive, wait: timeout, dead judge, the halves
adding up), on a local directory as the bucket with fake scorers; the proefrun; C2's scorer against naturalness.py's
classify path on a tiny model on 4 CPU devices."""

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))

import g4judge as gj  # noqa: E402


def samples(prefix, n_src, g=8, empty=(), long=()):
    rows = []
    for s in range(n_src):
        for j in range(g):
            uid = f"{prefix}{s}#{j}"
            hyp = "" if uid in empty else (" ".join(["lang"] * 1030) if uid in long else f"vertaling {s} {j} van de zin")
            rows.append({"uid": uid, "id": f"{prefix}{s}", "j": j, "source": f"source {s}", "hypothesis": hyp,
                         "token_ids": [5, 6, 7], "sampler_logp": [-0.1, -0.2, -0.3], "ended": bool(hyp),
                         "decode_loop": False, "english_left": False, "empty": not hyp, "tokens_out": 3})
    return rows


def put_shard(store, job, k, n, rows):
    data = gj.table_bytes(pa.Table.from_pylist(rows))
    store.write_once(f"{job}/samples/w{k}/shard-{n}.parquet", data)
    return {"rows": len(rows), "sha256": hashlib.sha256(data).hexdigest()}


def put_done(store, job, k, info):
    store.write_once(f"{job}/samples/w{k}/DONE", json.dumps(
        {"rows": sum(i["rows"] for i in info.values()), "shards": info}).encode())


def judges(store):
    out = [gj.Judge(store, "metricx", "w0", 0, gj.FakeMetricx()), gj.Judge(store, "metricx", "w2", 1, gj.FakeMetricx()),
           gj.Judge(store, "c2", "w1", None, gj.FakeC2())]
    for j in out:
        j.log = lambda m: None
        gj.Heartbeat(store, j.role, j.worker, j.part, j.scorer.name).set(state="idle")
    return out


def wait_args(root, job, writers, timeout=5.0, full=True, extra=()):
    a = ["wait", "--rl", str(root), "--job", job, "--writers", writers, "--timeout", str(timeout), "--poll", "0.05",
         "--judge", "metricx:w0:0", "--judge", "metricx:w2:1", "--judge", "c2:w1", *extra]
    return a + (["--full"] if full else [])


def test_part_of_is_fixed_and_halves():
    assert gj.part_of("x#0") == int(hashlib.sha256(b"x#0").hexdigest()[:8], 16) % 2
    parts = [gj.part_of(f"src{s}#{j}") for s in range(512) for j in range(8)]
    assert 0.45 < sum(parts) / len(parts) < 0.55


def test_local_store_writes_once_and_lists_no_temporaries(tmp_path):
    st = gj.LocalStore(tmp_path)
    st.write_once("a/b/x", b"1")
    st.write_once("a/b/x", b"1")  # the same bytes again (a retried write) is not an error
    with pytest.raises(gj.ObjectExists):
        st.write_once("a/b/x", b"2")
    (tmp_path / "a" / "b" / ".tmp-y").write_bytes(b"partial")
    st.overwrite("a/hb.json", b"{}")
    st.overwrite("a/hb.json", b"{1}")
    assert st.list("a/") == ["a/b/x", "a/hb.json"] and st.read("a/hb.json") == b"{1}"
    assert st.subdirs("a") == ["b"] and st.subdirs("none") == []


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data, if_generation_match=None, retry=None):
        assert retry is not None  # every upload retried (the heartbeat's too)
        from google.api_core.exceptions import PreconditionFailed

        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise PreconditionFailed("exists")
        self.bucket.objects[self.name] = data

    def download_as_bytes(self):
        return self.bucket.objects[self.name]

    def exists(self):
        return self.name in self.bucket.objects


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        return None


class FakePages(list):
    pass


class FakeClient:
    def __init__(self):
        self.b = FakeBucket()

    def bucket(self, name):
        assert name == "bkt"
        return self.b

    def list_blobs(self, bucket, prefix, delimiter=None):
        names = sorted(n for n in self.b.objects if n.startswith(prefix))
        pages = FakePages()
        pages.prefixes = set()
        for n in names:
            rest = n[len(prefix):]
            if delimiter and delimiter in rest:
                pages.prefixes.add(prefix + rest.split(delimiter)[0] + delimiter)
            else:
                pages.append(FakeBlob(self.b, n))
        return pages


def test_gcs_store_with_a_fake_client():
    pytest.importorskip("google.cloud.storage")
    st = gj.GcsStore("gs://bkt/rl/G4-7", client=FakeClient())
    st.write_once("iter/1/samples/w0/shard-0.parquet", b"x")
    st.write_once("iter/1/samples/w0/shard-0.parquet", b"x")
    with pytest.raises(gj.ObjectExists):
        st.write_once("iter/1/samples/w0/shard-0.parquet", b"y")
    st.write_once("iter/2/samples/w1/DONE", b"{}")
    assert st.client.b.objects["rl/G4-7/iter/1/samples/w0/shard-0.parquet"] == b"x"
    assert st.list("iter/1/") == ["iter/1/samples/w0/shard-0.parquet"]
    assert st.subdirs("iter") == ["1", "2"] and st.exists("iter/2/samples/w1/DONE")
    assert gj.jobs(st) == ["iter/1", "iter/2"]
    st.overwrite("judges/c2-w1.json", b"{}")
    st.overwrite("judges/c2-w1.json", b"{1}")
    assert st.read("judges/c2-w1.json") == b"{1}"


def test_done_and_sample_files_are_checked(tmp_path):
    st = gj.LocalStore(tmp_path)
    st.write_once("d1", json.dumps({"rows": 3, "shards": {"shard-0.parquet": {"rows": 2, "sha256": "0" * 64}}}).encode())
    with pytest.raises(gj.ProtocolError, match="not the sum"):
        gj.read_done(st, "d1")
    st.write_once("d2", json.dumps({"rows": 2, "shards": {"part-0.parquet": {"rows": 2, "sha256": "0" * 64}}}).encode())
    with pytest.raises(gj.ProtocolError, match="not a DONE"):
        gj.read_done(st, "d2")
    rows = samples("s", 2)
    with pytest.raises(gj.ProtocolError, match="no column"):
        gj.read_samples(gj.table_bytes(pa.Table.from_pylist([{k: v for k, v in r.items() if k != "token_ids"}
                                                              for r in rows])), "x")
    bad = [dict(r) for r in rows]
    bad[3]["uid"] = "s0#9"
    with pytest.raises(gj.ProtocolError, match="not <id>#<j>"):
        gj.read_samples(gj.table_bytes(pa.Table.from_pylist(bad)), "x")
    with pytest.raises(gj.ProtocolError, match="twice"):
        gj.read_samples(gj.table_bytes(pa.Table.from_pylist(rows + rows[:1])), "x")


def test_metricx_scores_its_half_and_leaves_an_empty_hypothesis_out():
    rows = samples("s", 16, empty={"s3#1", "s5#0"})
    seen = []

    class Rec(gj.FakeMetricx):
        def score(self, sources, hyps):
            seen.extend(hyps)
            return super().score(sources, hyps)

    out = {p: gj.score_rows("metricx", Rec(), rows, p) for p in (0, 1)}
    assert sorted(r["uid"] for p in (0, 1) for r in out[p]) == sorted(r["uid"] for r in rows)
    assert all(gj.part_of(r["uid"]) == p for p in (0, 1) for r in out[p])
    nulls = {r["uid"] for p in (0, 1) for r in out[p] if r["qe"] is None}
    assert nulls == {"s3#1", "s5#0"} and "" not in seen


def test_c2_never_truncates_and_marks_overflow():
    rows = samples("s", 4, long={"s1#2"}, empty={"s2#0"})
    passed = []

    class Rec(gj.FakeC2):
        def score(self, texts):
            passed.extend(texts)
            return super().score(texts)

    out = {r["uid"]: r for r in gj.score_rows("c2", Rec(), rows, None)}
    assert len(out) == len(rows)
    o = out["s1#2"]
    assert o["c2_overflow"] and o["c2_tokens"] == 1030 and o["p_c2"] is None and o["logit_c2"] is None
    assert all(len(t.split()) <= gj.C2_MAX_TOKENS for t in passed) and len(passed) == len(rows) - 1
    edge = samples("e", 1, g=2)
    edge[0]["hypothesis"] = " ".join(["w"] * gj.C2_MAX_TOKENS)
    edge[1]["hypothesis"] = " ".join(["w"] * (gj.C2_MAX_TOKENS + 1))
    e = gj.score_rows("c2", gj.FakeC2(), edge, None)
    assert [r["c2_overflow"] for r in e] == [False, True] and e[0]["p_c2"] is not None


def test_judge_streams_writes_done_last_and_is_idempotent(tmp_path):
    st = gj.LocalStore(tmp_path)
    j = gj.Judge(st, "c2", "w1", None, gj.FakeC2(), writers={"iter": [0]})
    j.log = lambda m: None
    info = {"shard-0.parquet": put_shard(st, "iter/1", 0, 0, samples("a", 4))}
    assert j.step()
    assert st.exists("iter/1/scores/c2/w1/shard-0-0.parquet") and not st.exists("iter/1/scores/c2/w1/DONE-0")
    info["shard-1.parquet"] = put_shard(st, "iter/1", 0, 1, samples("b", 4))
    put_done(st, "iter/1", 0, info)
    assert j.step()
    done = json.loads(st.read("iter/1/scores/c2/w1/DONE-0"))
    assert done["rows"] == done["samples_rows"] == 64
    assert {s: x["samples_sha256"] for s, x in done["shards"].items()} == {s: x["sha256"] for s, x in info.items()}
    before = {n: st.read(n) for n in st.list("iter/1/scores/")}
    assert "iter/1" in j.finished and not j.step()
    j2 = gj.Judge(st, "c2", "w1", None, gj.FakeC2())  # a restart: nothing to do, nothing rewritten
    j2.log = lambda m: None
    assert not j2.step()
    assert before == {n: st.read(n) for n in st.list("iter/1/scores/")}
    table = pq.read_table(io.BytesIO(st.read("iter/1/scores/c2/w1/shard-0-1.parquet")))
    assert table.schema.names == list(gj.SCORE_COLUMNS["c2"])
    meta = gj.read_score_meta(st.read("iter/1/scores/c2/w1/shard-0-1.parquet"), "x")
    assert meta["samples_sha256"] == info["shard-1.parquet"]["sha256"] and meta["rows"] == 32


def test_judge_refuses_a_shard_that_changed_after_it_was_scored(tmp_path):
    st = gj.LocalStore(tmp_path)
    j = gj.Judge(st, "metricx", "w0", 0, gj.FakeMetricx())
    j.log = lambda m: None
    info = {"shard-0.parquet": put_shard(st, "anchor", 0, 0, samples("a", 4, g=1))}
    j.step()
    info["shard-0.parquet"]["sha256"] = "f" * 64
    put_done(st, "anchor", 0, info)
    with pytest.raises(gj.ProtocolError, match="changed after it was scored"):
        j.step()
    assert not st.exists("anchor/scores/metricx/w0/DONE-0")


def test_judge_refuses_a_done_that_lists_other_shards(tmp_path):
    st = gj.LocalStore(tmp_path)
    j = gj.Judge(st, "c2", "w1", None, gj.FakeC2())
    j.log = lambda m: None
    info = {"shard-0.parquet": put_shard(st, "val/3", 3, 0, samples("v", 3, g=1))}
    put_done(st, "val/3", 3, {**info, "shard-7.parquet": {"rows": 1, "sha256": "0" * 64}})
    with pytest.raises(gj.ProtocolError, match="the directory has"):
        j.step()


def test_wait_passes_when_every_row_is_scored(tmp_path, capsys):
    st = gj.LocalStore(tmp_path)
    for k in range(4):
        info = {f"shard-{n}.parquet": put_shard(st, "iter/2", k, n, samples(f"w{k}n{n}-", 8)) for n in range(2)}
        put_done(st, "iter/2", k, info)
    for j in judges(st):
        j.step()
    assert gj.main(wait_args(tmp_path, "iter/2", "0,1,2,3")) == 0
    assert "SCORES_OK iter/2 rows=512" in capsys.readouterr().out
    assert gj.main(["alive", "--rl", str(tmp_path), "--judge", "metricx:w0:0", "--judge", "c2:w1"]) == 0


def test_wait_times_out_with_the_missing_shards(tmp_path, capsys):
    st = gj.LocalStore(tmp_path)
    put_done(st, "iter/1", 0, {"shard-0.parquet": put_shard(st, "iter/1", 0, 0, samples("a", 4))})
    mx = judges(st)
    for j in mx[:2]:
        j.step()  # C2's heartbeat is alive, but it scores nothing
    started = time.time()
    assert gj.main(wait_args(tmp_path, "iter/1", "0", timeout=0.3, full=False)) == 1
    out = capsys.readouterr().out
    assert "FAILED: not every score" in out and "c2:w1 iter/1/samples/w0/shard-0.parquet" in out
    assert time.time() - started < 5


def test_wait_fails_at_once_on_a_dead_judge_and_without_the_generation(tmp_path, capsys):
    st = gj.LocalStore(tmp_path)
    assert gj.main(wait_args(tmp_path, "iter/1", "0")) == 1
    assert "the generation has no" in capsys.readouterr().out
    put_done(st, "iter/1", 0, {"shard-0.parquet": put_shard(st, "iter/1", 0, 0, samples("a", 4))})
    judges(st)
    hb = json.loads(st.read("judges/c2-w1.json"))
    hb["epoch"] -= gj.HEARTBEAT_MAX + 1
    st.overwrite("judges/c2-w1.json", json.dumps(hb).encode())
    assert gj.main(wait_args(tmp_path, "iter/1", "0", timeout=60)) == 1
    out = capsys.readouterr().out
    assert "judges/c2-w1.json: heartbeat" in out and "missing" in out


def test_check_judges_states_parts_and_a_hung_shard(tmp_path):
    st = gj.LocalStore(tmp_path)
    hb = gj.Heartbeat(st, "metricx", "w0", 0, "fake")
    hb.set(state="loading")
    assert gj.check_judges(st, [("metricx", "w0", 0)]) == []
    assert "part 0, expected 1" in gj.check_judges(st, [("metricx", "w0", 1)])[0]
    hb.set(state="failed", error="RuntimeError: x")
    assert "state failed" in gj.check_judges(st, [("metricx", "w0", 0)])[0]
    hb.set(state="scoring", error=None, busy_since="2026-11-21T00:00:00+00:00", last_shard="iter/1/samples/w0/s")
    assert "for" in gj.check_judges(st, [("metricx", "w0", 0)], busy_max=60)[0]
    assert "no heartbeat" in gj.check_judges(st, [("c2", "w1", None)])[0]


def test_the_metricx_halves_must_add_up(tmp_path, capsys):
    st = gj.LocalStore(tmp_path)
    put_done(st, "iter/5", 1, {"shard-0.parquet": put_shard(st, "iter/5", 1, 0, samples("a", 8))})
    js = judges(st)
    for j in js:
        j.step()
    name = "iter/5/scores/metricx/w2/DONE-1"
    done = json.loads(st.read(name))
    done["shards"]["shard-0.parquet"]["rows"] -= 1
    (tmp_path / name).write_text(json.dumps(done))
    assert gj.main(wait_args(tmp_path, "iter/5", "1", full=False)) == 1
    assert "MetricX scored 63 rows of 64" in capsys.readouterr().out


def test_serve_stops_on_stop_and_keeps_a_heartbeat(tmp_path):
    st = gj.LocalStore(tmp_path)
    put_done(st, "iter/1", 0, {"shard-0.parquet": put_shard(st, "iter/1", 0, 0, samples("a", 2))})
    p = subprocess.Popen([sys.executable, str(CODE / "g4judge.py"), "serve", "--rl", str(tmp_path), "--role", "c2",
                          "--worker", "w1", "--scorer", "fake", "--poll", "0.1", "--heartbeat-every", "0.2"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = time.time() + 30
    while not st.exists("iter/1/scores/c2/w1/DONE-0") and time.time() < end:
        time.sleep(0.1)
    first = json.loads(st.read("judges/c2-w1.json"))["epoch"]
    time.sleep(0.6)
    assert json.loads(st.read("judges/c2-w1.json"))["epoch"] > first  # the thread beats while the judge idles
    st.write_once(gj.STOP, b"x")
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 0 and "STOPPED" in out
    assert json.loads(st.read("judges/c2-w1.json"))["state"] == "stopped"


def test_proefrun(tmp_path):
    r = subprocess.run([sys.executable, str(CODE / "g4judge.py"), "proef", "--dir", str(tmp_path / "bucket")],
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "JUDGE_PROEF_OK 12 checks" in r.stdout


C2_CASE = r"""
import sys, pickle, json, math
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/experiments/en-nl-translator")
import jax, jax.numpy as jnp, numpy as np
from flax import jax_utils
from functools import partial
from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.classification import ModernBertForSequenceClassification
import naturalness as nat
import g4judge as gj

out = Path(sys.argv[2])
config = MMBertConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                      local_attention=8, pad_token_id=0, num_labels=2, max_position_embeddings=1100)
model = ModernBertForSequenceClassification(config=config)
params = jax.jit(lambda r: model.init(r, jnp.ones((1, 8), jnp.int32), deterministic=True))(jax.random.PRNGKey(3))["params"]
from mmbjax.modules import rope
rope.clear_rotary_cache()  # params as load_pretrained_params gives them: without a trace (test_en_nl_naturalness.py)
d = config.to_dict()
d["num_labels"] = 2
(out / "config.json").write_text(json.dumps(d))
with open(out / "params.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)


class Tok:
    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None):
        ids = [[3 + (sum(map(ord, w)) % 50) for w in t.split()] for t in texts]
        ids = [[1] + i + [2] if add_special_tokens else i for i in ids]
        if truncation:
            ids = [i[:max_length] for i in ids]
        return {"input_ids": ids}


tok = Tok()
scorer = gj.C2Scorer(sys.argv[1], out, batch_size=2, tokenizer=tok)
texts = ["de kat zit op de mat", "een", "het huis is groot en wit en heeft een tuin"] * 3 + ["w " * 1022, "w " * 1023]
counts = scorer.count(texts)
assert counts[-2:] == [1022, 1023], counts[-2:]
rows = [{"uid": f"x{i}#0", "hypothesis": t} for i, t in enumerate(texts)]
scored = gj.score_rows("c2", scorer, rows, None)
assert scored[-1]["c2_overflow"] and scored[-1]["p_c2"] is None and not scored[-2]["c2_overflow"]
# naturalness.py classify's own path on the same texts (those that fit)
import finetune as ft
ids = tok(texts[:-1], truncation=True, max_length=1024)["input_ids"]
examples = [{"input_ids": i, "label": 0} for i in ids]
collate = partial(ft.collate_fn, pad_token_id=0, max_length=1024, pad_to_multiple_of=128)
ref = nat.predict(ft, nat.make_predict(model), jax_utils.replicate(params), examples, collate, 2 * ft.NUM_DEVICES)
dp = max(abs(a - s["p_c2"]) for a, s in zip(ref, scored))
dl = max(abs(math.log(s["p_c2"]) - math.log1p(-s["p_c2"]) - s["logit_c2"]) for s in scored[:-1])
assert dp <= 1e-6 and dl <= 1e-4, (dp, dl)
try:
    scorer.score(["w " * 1023])
    raise AssertionError("a text over 1,024 tokens with its specials was scored")
except RuntimeError as e:
    assert "truncated" in str(e)
print(f"C2_SCORER_OK dp={dp:.2e} dl={dl:.2e}")
"""


def test_c2_scorer_is_naturalness_classify_with_the_logit(tmp_path):
    pytest.importorskip("flax")
    pytest.importorskip("transformers")
    pytest.importorskip("optax")
    pytest.importorskip("datasets")
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=4")
    script = tmp_path / "case.py"
    script.write_text(C2_CASE)
    r = subprocess.run([sys.executable, str(script), str(ROOT), str(tmp_path)], capture_output=True, text=True,
                       env=env, timeout=600)
    assert r.returncode == 0 and "C2_SCORER_OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


def test_run_script_stages_dry(tmp_path):
    """run_g4arm7_judges.sh with DRY=1: each stage's job, with C1's training command on v2 and the judges' flags"""
    def dry(stage, **env):
        e = dict(os.environ, HOME=str(tmp_path), DRY="1", TAG="t1", **env)
        r = subprocess.run(["bash", str(CODE / "run_g4arm7_judges.sh"), stage], capture_output=True, text=True, env=e)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    conv = dry("convert")
    assert "--checkpoint gs://t5-dutch-english-usc2/mbert/v2-fw2optout-230m/ckpt_keep/1306000 --params-only" in conv
    train = dry("train")
    assert ("naturalness.py classifier --dumb-code" in train and "--checkpoint /dev/shm/translator/g4arm7/v2" in train
            and "--max-length 1024 --epochs 3 --learning-rate 5e-5 --seed 42 --batch-size 8" in train)
    assert "g4c2.py data7 --bt" in dry("data") and "--bt-validation" in dry("data")
    texts = dry("texts", CALIB="G4-6c=gs://b/t.parquet:gs://b/p.parquet")
    assert "gsutil -q cp gs://b/t.parquet" in texts and "gsutil -q cp gs://b/p.parquet" in texts
    assert "--calib G4-6c=" in texts and "g4c2.py cal7-texts" in texts
    assert "naturalness.py classify --dumb-code" in dry("classify")
    cal = dry("cal", CALIB="G4-6c=gs://b/t.parquet:gs://b/p.parquet")
    assert "g4c2.py cal7 --results" in cal and "--c1 G4-6c=" in cal
    mx = dry("serve", ROLE="metricx", PART="1")
    assert "g4judge.py serve --rl gs://t5-dutch-english-usc2/mbert/translator/sft/g4arm/rl/G4-7 --role metricx" in mx
    assert "--part 1 --root /dev/shm/translator/metricx" in mx and "--writers val=3" in mx
    c2 = dry("serve", ROLE="c2")
    assert "g4judge.py serve" in c2 and "--role c2" in c2 and "--dumb-code" in c2
