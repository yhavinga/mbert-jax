#!/usr/bin/env python3
"""The judges of arm G4-7 (protocol-g4.md §14, G4-7.4a): one persistent scorer per worker of mbert-mix that streams the
policy's samples from the bucket and writes their scores, and the chain's checks that every sample was scored.

    python g4judge.py serve --rl gs://.../rl/G4-7 --role metricx --worker w0 --part 0 --root DIR
    python g4judge.py serve --rl gs://.../rl/G4-7 --role c2 --worker w1 --dumb-code DIR --model DIR
    python g4judge.py alive --rl gs://.../rl/G4-7 --judge metricx:w0:0 --judge metricx:w2:1 --judge c2:w1
    python g4judge.py wait --rl gs://.../rl/G4-7 --job iter/3 --writers 0,1,2,3 --judge metricx:w0:0 ...
    python g4judge.py proef --dir DIR     (the proefrun: a local directory as the bucket, fake scorers, no model)

--rl is a gs:// directory (the Python client of GCS) or a local directory (the fake bucket of the proefrun and the
tests). A job is iter/<i>, val/<i> or anchor. Under it a generating worker k writes samples/w<k>/shard-<n>.parquet and,
last, samples/w<k>/DONE; a judge writes scores/<role>/w<m>/shard-<k>-<n>.parquet as soon as a shard appears (streaming)
and scores/<role>/w<m>/DONE-<k> once every shard of that DONE is scored, with the sha256 of the sample file it read.
A shard with a score is skipped (restartable, idempotent); an object is written once (GCS if_generation_match=0, a hard
link locally), except the heartbeat judges/<role>-w<m>.json, which a thread rewrites every 60 s. judges/STOP ends a
judge after its shard. MetricX is split over two workers on the parity of sha256(uid) (part 0 and 1); C2 scores every
row, counts its tokens without the special tokens and never truncates: more than 1,022 is c2_overflow (no score).

serve --scorer fake writes the same files with deterministic stand-in scores (hashes of the texts), for the proefrun.
"""

import argparse
import hashlib
import io
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("metricx", "c2")
SAMPLE_COLUMNS = ("uid", "id", "j", "source", "hypothesis", "token_ids", "sampler_logp", "ended", "decode_loop",
                  "english_left", "empty", "tokens_out")
SCORE_COLUMNS = {"metricx": ("uid", "qe"), "c2": ("uid", "p_c2", "logit_c2", "c2_tokens", "c2_overflow")}
C2_MAX_TOKENS = 1022  # Yeb 21 Nov: more than 1,022 tokens of C2's tokenizer (no special tokens) is a fail
C2_MAX_LENGTH = 1024  # the classifier's --max-length, as C1
HEARTBEAT_EVERY = 60
HEARTBEAT_MAX = 300  # the chain: a heartbeat older than 5 min is FAILED
SCORE_TIMEOUT = 900  # every score within 15 min of the generation's last DONE
BUSY_MAX = 900  # one shard longer than this is a hung judge
STOP = "judges/STOP"
META_KEY = b"g4judge"
SHARD_RE = re.compile(r"^shard-(\d+)\.parquet$")
WORKER_RE = re.compile(r"^w(\d+)$")


class ObjectExists(Exception):
    """A write-once object exists with other content"""


class ProtocolError(Exception):
    """The bucket's files do not follow G4-7.4a"""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def part_of(uid):
    """MetricX's half of a sample: the parity of the first 8 hex digits of sha256(uid) (w0 part 0, w2 part 1)"""
    return int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16) % 2


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------- stores

class LocalStore:
    """A directory as the bucket. Write-once by a hard link of a finished temporary file, so a reader never sees a
    partial object, as GCS writes an object atomically."""

    def __init__(self, root):
        self.root = Path(root)

    def __str__(self):
        return str(self.root)

    def list(self, prefix):
        base = self.root / prefix
        if not base.is_dir():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*")
                      if p.is_file() and not p.name.startswith(".tmp-"))

    def subdirs(self, prefix):
        base = self.root / prefix
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    def exists(self, name):
        return (self.root / name).is_file()

    def read(self, name):
        return (self.root / name).read_bytes()

    def mtime(self, name):
        return (self.root / name).stat().st_mtime

    def _tmp(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp-{path.name}-{os.getpid()}-{uuid.uuid4().hex}"
        tmp.write_bytes(data)
        return tmp

    def write_once(self, name, data):
        path = self.root / name
        tmp = self._tmp(path, data)
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ObjectExists(name) from None
        finally:
            tmp.unlink()

    def overwrite(self, name, data):
        path = self.root / name
        os.replace(self._tmp(path, data), path)


def split_gs(url):
    if not url.startswith("gs://"):
        raise ValueError(f"not a gs:// URL: {url}")
    bucket, _, name = url[len("gs://"):].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url}")
    return bucket, name.strip("/")


class GcsStore:
    """A gs:// directory through the Python client of GCS (not gsutil: its start-up per call, and on the TPU images a
    snap that refuses to start under no_new_privs)."""

    def __init__(self, url, client=None):
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        self.client = client
        self.bucket_name, self.prefix = split_gs(url)
        self.bucket = client.bucket(self.bucket_name)
        self.url = url

    def __str__(self):
        return self.url

    def _full(self, name):
        return f"{self.prefix}/{name}" if self.prefix else name

    def _rel(self, full):
        return full[len(self.prefix) + 1:] if self.prefix else full

    def list(self, prefix):
        full = self._full(prefix.rstrip("/")) + "/"
        return sorted(self._rel(b.name) for b in self.client.list_blobs(self.bucket_name, prefix=full))

    def subdirs(self, prefix):
        full = self._full(prefix.rstrip("/")) + "/"
        pages = self.client.list_blobs(self.bucket_name, prefix=full, delimiter="/")
        list(pages)  # the prefixes are filled while the pages are read
        return sorted(p[len(full):].rstrip("/") for p in pages.prefixes)

    def exists(self, name):
        return self.bucket.blob(self._full(name)).exists()

    def read(self, name):
        return self.bucket.blob(self._full(name)).download_as_bytes()

    def mtime(self, name):
        blob = self.bucket.get_blob(self._full(name))
        if blob is None:
            raise FileNotFoundError(name)
        return blob.updated.timestamp()

    def write_once(self, name, data):
        from google.api_core.exceptions import PreconditionFailed

        blob = self.bucket.blob(self._full(name))
        try:
            blob.upload_from_string(data, if_generation_match=0)
        except PreconditionFailed:
            # a retried upload that had succeeded meets its own object: the same bytes are not an error
            if self.read(name) != data:
                raise ObjectExists(name) from None

    def overwrite(self, name, data):
        self.bucket.blob(self._full(name)).upload_from_string(data)


def open_store(rl):
    return GcsStore(rl) if rl.startswith("gs://") else LocalStore(rl)


# ---------------------------------------------------------------------------------------------------- protocol

def sample_dir(job, k):
    return f"{job}/samples/w{k}"


def score_dir(job, role, worker):
    return f"{job}/scores/{role}/{worker}"


def score_name(job, role, worker, k, n):
    return f"{score_dir(job, role, worker)}/shard-{k}-{n}.parquet"


def score_done_name(job, role, worker, k):
    return f"{score_dir(job, role, worker)}/DONE-{k}"


def heartbeat_name(role, worker):
    return f"judges/{role}-{worker}.json"


def jobs(store):
    out = []
    for kind in ("iter", "val"):
        out += [f"{kind}/{d}" for d in store.subdirs(kind) if d.isdigit()]
    if "samples" in store.subdirs("anchor"):
        out.append("anchor")
    return out


def job_samples(store, job):
    """{k: {"shards": {n: name}, "done": name or None}} of a job's samples directory"""
    out = {}
    base = f"{job}/samples/"
    for name in store.list(base):
        rest = name[len(base):].split("/")
        if len(rest) != 2 or not WORKER_RE.match(rest[0]):
            raise ProtocolError(f"{name}: not samples/w<k>/shard-<n>.parquet or samples/w<k>/DONE")
        k = int(WORKER_RE.match(rest[0]).group(1))
        entry = out.setdefault(k, {"shards": {}, "done": None})
        if rest[1] == "DONE":
            entry["done"] = name
        elif SHARD_RE.match(rest[1]):
            entry["shards"][int(SHARD_RE.match(rest[1]).group(1))] = name
        else:
            raise ProtocolError(f"{name}: not samples/w<k>/shard-<n>.parquet or samples/w<k>/DONE")
    return out


def read_done(store, name):
    """A generating worker's DONE: {"rows": int, "shards": {"shard-<n>.parquet": {"rows": int, "sha256": hex}}}"""
    try:
        done = json.loads(store.read(name))
        shards = done["shards"]
        if not isinstance(done["rows"], int) or not isinstance(shards, dict) or not shards:
            raise ValueError("rows not an int, or no shards")
        for shard, info in shards.items():
            if not SHARD_RE.match(shard) or not isinstance(info["rows"], int) or len(info["sha256"]) != 64:
                raise ValueError(f"shard {shard}: {info}")
    except (ValueError, KeyError, TypeError) as e:
        raise ProtocolError(f"{name}: not a DONE of G4-7.4a ({e})") from None
    if sum(i["rows"] for i in shards.values()) != done["rows"]:
        raise ProtocolError(f"{name}: rows {done['rows']} is not the sum over its shards")
    return done


def read_samples(data, name):
    """The rows a judge reads from a sample shard (uid, source, hypothesis), after checking its columns and uids"""
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(data))
    missing = [c for c in SAMPLE_COLUMNS if c not in table.schema.names]
    if missing:
        raise ProtocolError(f"{name}: no column(s) {missing} (G4-7.4a)")
    rows = table.select(["uid", "id", "j", "source", "hypothesis"]).to_pylist()
    uids = [r["uid"] for r in rows]
    if len(set(uids)) != len(uids):
        raise ProtocolError(f"{name}: a uid twice")
    for r in rows:
        if r["uid"] != f"{r['id']}#{r['j']}":
            raise ProtocolError(f"{name}: uid {r['uid']!r} is not <id>#<j> ({r['id']!r}, {r['j']!r})")
        if not isinstance(r["source"], str) or not r["source"]:
            raise ProtocolError(f"{name}: {r['uid']} has no source")
        if not isinstance(r["hypothesis"], str):
            raise ProtocolError(f"{name}: {r['uid']} has no hypothesis string (an empty one is '')")
    return rows


def score_table(role, rows, meta):
    import pyarrow as pa

    types = {"uid": pa.string(), "qe": pa.float64(), "p_c2": pa.float64(), "logit_c2": pa.float64(),
             "c2_tokens": pa.int64(), "c2_overflow": pa.bool_()}
    schema = pa.schema([(c, types[c]) for c in SCORE_COLUMNS[role]], metadata={META_KEY: json.dumps(meta).encode()})
    return pa.Table.from_pylist(rows, schema=schema)


def table_bytes(table):
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def read_score_meta(data, name):
    import pyarrow.parquet as pq

    meta = pq.read_schema(io.BytesIO(data)).metadata or {}
    if META_KEY not in meta:
        raise ProtocolError(f"{name}: no {META_KEY.decode()} metadata")
    return json.loads(meta[META_KEY])


# ---------------------------------------------------------------------------------------------------- scorers

class FakeMetricx:
    """Deterministic stand-in for MetricX-24 QE (0..10 from a hash of source and hypothesis), for the proefrun"""
    name = "fake-metricx"

    def __init__(self, delay=0.0):
        self.delay = delay

    def score(self, sources, hypotheses):
        time.sleep(self.delay)
        return [int(sha256_bytes(f"{s}\0{h}".encode())[:6], 16) % 100000 / 10000 for s, h in zip(sources, hypotheses)]


class FakeC2:
    """Deterministic stand-in for C2: a token per white-space word and a logit from a hash of the text"""
    name = "fake-c2"

    def __init__(self, delay=0.0):
        self.delay = delay

    def count(self, texts):
        return [len(t.split()) for t in texts]

    def score(self, texts):
        import math

        time.sleep(self.delay)
        out = []
        for t in texts:
            logit = (int(sha256_bytes(t.encode())[:6], 16) % 16000 - 8000) / 1000
            out.append((1 / (1 + math.exp(-logit)), logit))
        return out


class MetricxScorer:
    """MetricX-24 hybrid XXL v2p6 bf16 in QE mode: metricx_qe.py's Scorer, scores rounded to 4 decimals as its score"""
    name = "metricx-24-hybrid-xxl-v2p6-bfloat16"

    def __init__(self, root):
        import metricx_qe

        self.scorer = metricx_qe.Scorer("xxl", root, "bfloat16")

    def score(self, sources, hypotheses):
        scores, _ = self.scorer.score(list(sources), list(hypotheses))
        return [round(float(s), 4) for s in scores]


class C2Scorer:
    """C2 as naturalness.py classify runs it (the DUMB code's load_pretrained_params, collate_fn and shard_batch, the
    classification head, the rotary cache warmed at the maximum length, the softmax in float32), on the 4 chips of one
    host, with logit_1 - logit_0 beside P(label 1). No text is truncated: one over C2_MAX_TOKENS is not scored."""
    name = "c2"

    def __init__(self, dumb_code, model_dir, batch_size=8, pad_to_multiple_of=128, tokenizer=None):
        from functools import partial

        import naturalness as nat

        nat.dumb_paths(dumb_code)
        import finetune as ft
        import jax
        from flax import jax_utils

        if (jax.device_count(), jax.process_count()) != (4, 1):
            raise SystemExit(f"{jax.device_count()} devices, {jax.process_count()} processes: one host of 4 chips")
        params, config = ft.load_pretrained_params(Path(model_dir))
        if getattr(config, "num_labels", None) != 2:
            raise SystemExit(f"{model_dir}: num_labels {getattr(config, 'num_labels', None)}, not the classifier")
        from mmbjax.heads.classification import ModernBertForSequenceClassification

        model = ModernBertForSequenceClassification(config=config)
        nat.warm_rotary_cache(model, params, C2_MAX_LENGTH)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or "yhavinga/dutch-llama-tokenizer")
        self.ft, self.tokenizer = ft, tokenizer
        self.params = jax_utils.replicate(params)
        self.collate = partial(ft.collate_fn, pad_token_id=tokenizer.pad_token_id, max_length=C2_MAX_LENGTH,
                               pad_to_multiple_of=pad_to_multiple_of)
        self.total = batch_size * ft.NUM_DEVICES

        def heads(p, input_ids, attention_mask):
            logits = model.apply({"params": p}, input_ids, attention_mask=attention_mask,
                                 deterministic=True).logits.astype("float32")
            # P(label 1) exactly as naturalness.make_predict computes it
            return jax.nn.softmax(logits, axis=-1)[..., 1], logits[..., 1] - logits[..., 0]

        self.fn = jax.pmap(heads, axis_name="batch")

    def count(self, texts):
        return [len(i) for i in self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    def score(self, texts):
        import numpy as np

        ids = self.tokenizer(list(texts))["input_ids"]
        longest = max(len(i) for i in ids)
        if longest > C2_MAX_LENGTH:
            raise RuntimeError(f"a text of {longest} tokens with its special tokens reached the scorer "
                               f"(more than {C2_MAX_LENGTH}): it would be truncated")
        examples = [{"input_ids": i, "label": 0} for i in ids]
        out = []
        for i in range(0, len(examples), self.total):
            chunk = examples[i:i + self.total]
            filled = chunk + [examples[0]] * (self.total - len(chunk))
            batch = self.ft.shard_batch(self.collate(filled))
            p, logit = self.fn(self.params, batch["input_ids"], batch["attention_mask"])
            p, logit = np.asarray(p).reshape(-1), np.asarray(logit).reshape(-1)
            out += [(float(a), float(b)) for a, b in zip(p[:len(chunk)], logit[:len(chunk)])]
        return out


def score_rows(role, scorer, rows, part):
    """The score rows of a shard: MetricX for its part only (null for an empty hypothesis, as metricx_qe.py score
    leaves an empty candidate out), C2 for every row (null p and logit when c2_overflow)"""
    if role == "metricx":
        mine = [r for r in rows if part_of(r["uid"]) == part]
        todo = [r for r in mine if r["hypothesis"]]
        qe = dict(zip((r["uid"] for r in todo), scorer.score([r["source"] for r in todo],
                                                             [r["hypothesis"] for r in todo]) if todo else []))
        return [{"uid": r["uid"], "qe": qe.get(r["uid"])} for r in mine]
    tokens = scorer.count([r["hypothesis"] for r in rows]) if rows else []
    fits = [r for r, n in zip(rows, tokens) if n <= C2_MAX_TOKENS]
    scored = dict(zip((r["uid"] for r in fits), scorer.score([r["hypothesis"] for r in fits]) if fits else []))
    out = []
    for r, n in zip(rows, tokens):
        p, logit = scored.get(r["uid"], (None, None))
        out.append({"uid": r["uid"], "p_c2": p, "logit_c2": logit, "c2_tokens": n, "c2_overflow": n > C2_MAX_TOKENS})
    return out


def make_scorer(args):
    if args.scorer == "fake":
        return (FakeMetricx if args.role == "metricx" else FakeC2)(args.fake_delay)
    if args.role == "metricx":
        if not args.root:
            raise SystemExit("--role metricx needs --root (where MetricX-24 is downloaded)")
        return MetricxScorer(args.root)
    if not args.dumb_code or not args.model:
        raise SystemExit("--role c2 needs --dumb-code and --model (C2's directory: params.pkl, config.json)")
    return C2Scorer(args.dumb_code, args.model, args.batch_size)


# ---------------------------------------------------------------------------------------------------- serve

class Heartbeat:
    """judges/<role>-<worker>.json, rewritten by a thread every `every` s while the process lives (also while a shard
    is scored or a model loads), and at once on a change of state"""

    def __init__(self, store, role, worker, part, scorer_name, every=HEARTBEAT_EVERY):
        self.store, self.every = store, every
        self.lock = threading.Lock()
        self.info = {"pid": os.getpid(), "host": socket.gethostname(), "role": role, "worker": worker, "part": part,
                     "scorer": scorer_name, "started": utc_now(), "state": "starting", "busy_since": None,
                     "last_shard": None, "scored_rows": 0, "scored_shards": 0, "error": None}
        self.name = heartbeat_name(role, worker)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def set(self, **changes):
        with self.lock:
            self.info.update(changes)
        self.write()

    def write(self):
        with self.lock:
            now = time.time()
            body = dict(self.info, time=utc_now(), epoch=round(now, 3))
        self.store.overwrite(self.name, (json.dumps(body) + "\n").encode())

    def _run(self):
        while not self.stop.wait(self.every):
            try:
                self.write()
            except Exception as e:  # the main loop reports its own failures; a missed beat shows as an old heartbeat
                print(f"HEARTBEAT write failed: {e!r}", file=sys.stderr, flush=True)

    def start(self):
        self.write()
        self.thread.start()

    def close(self, state, error=None):
        self.stop.set()
        self.set(state=state, busy_since=None, error=error)


class Judge:
    def __init__(self, store, role, worker, part, scorer, writers=None, log=print):
        if role not in ROLES:
            raise SystemExit(f"--role {role}: one of {ROLES}")
        if (role == "metricx") != (part is not None):
            raise SystemExit("--part 0 or 1 for metricx, none for c2")
        self.store, self.role, self.worker, self.part, self.scorer = store, role, worker, part, scorer
        self.writers = writers or {}
        self.finished = set()
        self.meta = {}  # score name -> its metadata, for the shards this process scored
        self.log = log
        self.heartbeat = None

    def score_shard(self, job, k, n, name):
        data = self.store.read(name)
        rows = read_samples(data, name)
        started = time.time()
        out = score_rows(self.role, self.scorer, rows, self.part)
        meta = {"samples": name, "samples_sha256": sha256_bytes(data), "samples_rows": len(rows), "rows": len(out),
                "role": self.role, "worker": self.worker, "part": self.part, "scorer": self.scorer.name,
                "seconds": round(time.time() - started, 3), "time": utc_now()}
        target = score_name(job, self.role, self.worker, k, n)
        self.store.write_once(target, table_bytes(score_table(self.role, out, meta)))
        self.meta[target] = meta
        extra = ""
        if self.role == "c2":
            extra = f" overflow={sum(r['c2_overflow'] for r in out)}"
        self.log(f"SCORED {target} rows={len(out)}/{len(rows)} {meta['seconds']} s{extra}")
        return len(out)

    def shard_meta(self, target):
        if target not in self.meta:
            self.meta[target] = read_score_meta(self.store.read(target), target)
        return self.meta[target]

    def write_done(self, job, k, done_name, shards):
        done = read_done(self.store, done_name)
        listed = {int(SHARD_RE.match(s).group(1)) for s in done["shards"]}
        if listed != set(shards):
            raise ProtocolError(f"{done_name}: shards {sorted(listed)}, the directory has {sorted(shards)}")
        out = {"rows": 0, "samples_rows": 0, "shards": {}, "role": self.role, "worker": self.worker,
               "part": self.part, "time": utc_now()}
        for shard, info in sorted(done["shards"].items()):
            n = int(SHARD_RE.match(shard).group(1))
            target = score_name(job, self.role, self.worker, k, n)
            meta = self.shard_meta(target)
            if meta["samples_sha256"] != info["sha256"] or meta["samples_rows"] != info["rows"]:
                raise ProtocolError(f"{target}: scored {meta['samples_rows']} rows of sha256 "
                                    f"{meta['samples_sha256'][:12]}, {done_name} has {info['rows']} of "
                                    f"{info['sha256'][:12]}: the shard changed after it was scored")
            out["shards"][shard] = {"rows": meta["rows"], "samples_rows": meta["samples_rows"],
                                    "samples_sha256": meta["samples_sha256"]}
            out["rows"] += meta["rows"]
            out["samples_rows"] += meta["samples_rows"]
        target = score_done_name(job, self.role, self.worker, k)
        self.store.write_once(target, (json.dumps(out, sort_keys=True) + "\n").encode())
        self.log(f"DONE {target} rows={out['rows']} of {out['samples_rows']}")

    def step(self):
        """One pass over the jobs: every new shard scored, every complete worker's marker written. True if it did
        anything."""
        did = False
        for job in jobs(self.store):
            if job in self.finished:
                continue
            samples = job_samples(self.store, job)
            mine = set(self.store.list(score_dir(job, self.role, self.worker) + "/"))
            for k in sorted(samples):
                for n in sorted(samples[k]["shards"]):
                    target = score_name(job, self.role, self.worker, k, n)
                    if target in mine:
                        continue
                    if self.store.exists(STOP):
                        return did
                    name = samples[k]["shards"][n]
                    if self.heartbeat:
                        self.heartbeat.set(state="scoring", busy_since=utc_now(), last_shard=name)
                    rows = self.score_shard(job, k, n, name)
                    mine.add(target)
                    if self.heartbeat:
                        with self.heartbeat.lock:
                            self.heartbeat.info["scored_rows"] += rows
                            self.heartbeat.info["scored_shards"] += 1
                        self.heartbeat.set(state="idle", busy_since=None)
                    did = True
                done_target = score_done_name(job, self.role, self.worker, k)
                if samples[k]["done"] and done_target not in mine:
                    self.write_done(job, k, samples[k]["done"], samples[k]["shards"])
                    mine.add(done_target)
                    did = True
            kind = job.split("/")[0]
            expected = self.writers.get(kind)
            done_ks = {k for k in samples if score_done_name(job, self.role, self.worker, k) in mine}
            if expected is not None and done_ks == set(expected):
                self.finished.add(job)
        return did


def parse_writers(values):
    """kind=k,k,...: the generating workers of a kind of job (iter=0,1,2,3 val=3 anchor=0,1,2,3)"""
    out = {}
    for v in values or []:
        kind, _, ks = v.partition("=")
        if kind not in ("iter", "val", "anchor") or not ks:
            raise SystemExit(f"--writers {v!r}: iter=0,1,2,3, val=3 or anchor=0,1,2,3")
        out[kind] = [int(k) for k in ks.split(",")]
    return out


def stop_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def serve(args):
    import signal

    signal.signal(signal.SIGTERM, stop_on_signal)
    store = open_store(args.rl)
    part = args.part if args.role == "metricx" else None
    judge = Judge(store, args.role, args.worker, part, None, parse_writers(args.writers),
                  log=lambda m: print(f"{utc_now()} {m}", flush=True))
    hb = Heartbeat(store, args.role, args.worker, part, f"{args.scorer}", every=args.heartbeat_every)
    judge.heartbeat = hb
    hb.info["state"] = "loading"
    hb.start()
    try:
        judge.scorer = make_scorer(args)
        hb.set(state="idle", scorer=judge.scorer.name)
        print(f"{utc_now()} SERVE {args.role} {args.worker} part={part} on {store} scorer={judge.scorer.name}",
              flush=True)
        last_work = time.time()
        while True:
            if store.exists(STOP):
                hb.close("stopped")
                print(f"{utc_now()} STOPPED by {STOP}", flush=True)
                return 0
            if judge.step():
                last_work = time.time()
                continue
            if args.max_idle and time.time() - last_work > args.max_idle:
                hb.close("stopped-idle")
                print(f"{utc_now()} STOPPED after {args.max_idle} s without work", flush=True)
                return 0
            time.sleep(args.poll)
    except SystemExit as e:
        hb.close("stopped-signal" if e.code == 128 + 15 else "failed", error=f"SystemExit: {e.code}"[:2000])
        raise
    except BaseException as e:
        hb.close("failed", error=f"{type(e).__name__}: {e}"[:2000])
        raise


# ---------------------------------------------------------------------------------------------------- the chain's checks

def parse_judge(value):
    """role:w<m>[:part]"""
    parts = value.split(":")
    if len(parts) not in (2, 3) or parts[0] not in ROLES or not WORKER_RE.match(parts[1]):
        raise SystemExit(f"--judge {value!r}: metricx:w0:0, metricx:w2:1 or c2:w1")
    part = int(parts[2]) if len(parts) == 3 else None
    if (parts[0] == "metricx") != (part in (0, 1)):
        raise SystemExit(f"--judge {value!r}: metricx needs part 0 or 1, c2 none")
    return parts[0], parts[1], part


def check_judges(store, judges, now=None, heartbeat_max=HEARTBEAT_MAX, busy_max=BUSY_MAX):
    """The problems of the judges' heartbeats: missing, older than heartbeat_max, a state other than loading, idle or
    scoring, or one shard for longer than busy_max"""
    now = time.time() if now is None else now
    problems = []
    for role, worker, part in judges:
        name = heartbeat_name(role, worker)
        if not store.exists(name):
            problems.append(f"{name}: no heartbeat")
            continue
        hb = json.loads(store.read(name))
        age = now - hb["epoch"]
        if age > heartbeat_max:
            problems.append(f"{name}: heartbeat {age:.0f} s old (pid {hb['pid']} on {hb['host']}, {hb['state']})")
        elif hb["state"] not in ("loading", "idle", "scoring"):
            problems.append(f"{name}: state {hb['state']} ({hb.get('error')})")
        elif hb["part"] != part:
            problems.append(f"{name}: part {hb['part']}, expected {part}")
        elif hb["state"] == "scoring" and hb.get("busy_since"):
            busy = now - datetime.fromisoformat(hb["busy_since"]).timestamp()
            if busy > busy_max:
                problems.append(f"{name}: on {hb['last_shard']} for {busy:.0f} s")
    return problems


def scores_state(store, job, writers, judges):
    """(missing, problems): the shards each judge has not scored yet, and what is wrong with the markers it wrote"""
    missing, problems = [], []
    dones = {k: read_done(store, f"{sample_dir(job, k)}/DONE") for k in writers}
    for role, worker, part in judges:
        for k in writers:
            marker = score_done_name(job, role, worker, k)
            if not store.exists(marker):
                have = set(store.list(score_dir(job, role, worker) + "/"))
                missing += [f"{role}:{worker} {sample_dir(job, k)}/{s}" for s in sorted(dones[k]["shards"])
                            if score_name(job, role, worker, k, int(SHARD_RE.match(s).group(1))) not in have]
                if not any(m.startswith(f"{role}:{worker} {sample_dir(job, k)}/") for m in missing):
                    missing.append(f"{role}:{worker} {marker} (every shard scored, no marker)")
                continue
            got = json.loads(store.read(marker))
            if got.get("part") != part:
                problems.append(f"{marker}: part {got.get('part')}, expected {part}")
            if sorted(got["shards"]) != sorted(dones[k]["shards"]):
                problems.append(f"{marker}: shards {sorted(got['shards'])}, DONE has {sorted(dones[k]['shards'])}")
                continue
            for shard, info in dones[k]["shards"].items():
                mine = got["shards"][shard]
                if mine["samples_sha256"] != info["sha256"] or mine["samples_rows"] != info["rows"]:
                    problems.append(f"{marker}: {shard} scored from other bytes than DONE's")
                if role == "c2" and mine["rows"] != info["rows"]:
                    problems.append(f"{marker}: {shard} {mine['rows']} rows scored of {info['rows']}")
    return missing, problems, dones


def metricx_complete(store, job, writers, judges, dones):
    """The two MetricX halves together have every row of every shard (each judge scored exactly its own part,
    checked when it scored)"""
    problems = []
    mx = [(w, p) for r, w, p in judges if r == "metricx"]
    if sorted(p for _, p in mx) != [0, 1]:
        problems.append(f"MetricX judges {mx}: not one of part 0 and one of part 1")
        return problems
    for k in writers:
        markers = [json.loads(store.read(score_done_name(job, "metricx", w, k))) for w, _ in mx]
        for shard, info in dones[k]["shards"].items():
            total = sum(m["shards"][shard]["rows"] for m in markers)
            if total != info["rows"]:
                problems.append(f"{sample_dir(job, k)}/{shard}: MetricX scored {total} rows of {info['rows']}")
    return problems


def full_uid_check(store, job, writers, judges, dones):
    """The uids themselves (reads every sample and score file: run it in the bucket's region)"""
    import pyarrow.parquet as pq

    problems = []
    samples = set()
    for k in writers:
        for shard in dones[k]["shards"]:
            samples |= set(pq.read_table(io.BytesIO(store.read(f"{sample_dir(job, k)}/{shard}")),
                                         columns=["uid"]).column("uid").to_pylist())
    by_role = {}
    for role, worker, part in judges:
        for name in store.list(score_dir(job, role, worker) + "/"):
            if name.endswith(".parquet"):
                uids = pq.read_table(io.BytesIO(store.read(name)), columns=["uid"]).column("uid").to_pylist()
                if part is not None and any(part_of(u) != part for u in uids):
                    problems.append(f"{name}: a uid of the other part")
                seen = by_role.setdefault(role, [])
                seen += uids
    for role, uids in by_role.items():
        if len(uids) != len(set(uids)):
            problems.append(f"{role}: {len(uids) - len(set(uids))} uids scored twice")
        if set(uids) != samples:
            problems.append(f"{role}: {len(samples - set(uids))} uids of the samples not scored, "
                            f"{len(set(uids) - samples)} scored uids not among the samples")
    return problems


def alive(args):
    store = open_store(args.rl)
    problems = check_judges(store, [parse_judge(j) for j in args.judge], heartbeat_max=args.heartbeat_max,
                            busy_max=args.busy_max)
    if problems:
        print("FAILED: " + "; ".join(problems), flush=True)
        return 1
    print(f"JUDGES_ALIVE {len(args.judge)}", flush=True)
    return 0


def wait(args):
    """After the generation of a job: every DONE of the writers is there; then every judge's DONE-<k> within
    --timeout s of the last DONE, the heartbeats alive meanwhile, and every row scored"""
    store = open_store(args.rl)
    judges = [parse_judge(j) for j in args.judge]
    writers = [int(k) for k in args.writers.split(",")]
    absent = [f"{sample_dir(args.job, k)}/DONE" for k in writers if not store.exists(f"{sample_dir(args.job, k)}/DONE")]
    if absent:
        print(f"FAILED: the generation has no {absent}", flush=True)
        return 1
    last_done = max(store.mtime(f"{sample_dir(args.job, k)}/DONE") for k in writers)
    while True:
        dead = check_judges(store, judges, heartbeat_max=args.heartbeat_max, busy_max=args.busy_max)
        missing, problems, dones = scores_state(store, args.job, writers, judges)
        if problems:
            print("FAILED: " + "; ".join(problems), flush=True)
            return 1
        if not missing:
            break
        if dead:
            print("FAILED: " + "; ".join(dead) + f"; missing {len(missing)}: " + "; ".join(missing[:20]), flush=True)
            return 1
        if time.time() - last_done > args.timeout:
            print(f"FAILED: not every score {args.timeout} s after the last DONE; missing {len(missing)}: "
                  + "; ".join(missing), flush=True)
            return 1
        time.sleep(args.poll)
    problems = metricx_complete(store, args.job, writers, judges, dones)
    if not problems and args.full:
        problems = full_uid_check(store, args.job, writers, judges, dones)
    if problems:
        print("FAILED: " + "; ".join(problems), flush=True)
        return 1
    rows = sum(d["rows"] for d in dones.values())
    print(f"SCORES_OK {args.job} rows={rows} writers={writers} judges={len(judges)} "
          f"after_last_done={time.time() - last_done:.0f}s{' full' if args.full else ''}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- proefrun

def fake_samples(job_seed, sources, g):
    """Rows of the sample schema for sources x g, with empties and a long hypothesis among them"""
    import random

    rnd = random.Random(job_seed)
    rows = []
    for s, src in sources:
        for j in range(g):
            words = rnd.randint(1, 30)
            hyp = " ".join(rnd.choice(("de", "het", "een", "huis", "vertaling", "natuurlijk", "zin", "Nederlands"))
                           for _ in range(words))
            if rnd.random() < 0.03:
                hyp = ""
            if rnd.random() < 0.02:
                hyp = " ".join(["lang"] * (C2_MAX_TOKENS + 5))
            ids = [rnd.randint(3, 1000) for _ in range(max(1, len(hyp.split())))]
            rows.append({"uid": f"{s}#{j}", "id": s, "j": j, "source": src, "hypothesis": hyp, "token_ids": ids,
                         "sampler_logp": [-rnd.random() for _ in ids], "ended": hyp != "", "decode_loop": False,
                         "english_left": False, "empty": hyp == "", "tokens_out": len(ids)})
    return rows


def write_sample_shards(store, job, k, rows, per_shard, pause=0.0, done=True, log=print):
    """A generating worker as G4-7.4a has it: the shards, then DONE"""
    import pyarrow as pa

    info = {}
    for n, i in enumerate(range(0, len(rows), per_shard)):
        data = table_bytes(pa.Table.from_pylist(rows[i:i + per_shard]))
        name = f"shard-{n}.parquet"
        store.write_once(f"{sample_dir(job, k)}/{name}", data)
        info[name] = {"rows": len(rows[i:i + per_shard]), "sha256": sha256_bytes(data)}
        log(f"GEN {job} w{k} {name} rows={info[name]['rows']}")
        time.sleep(pause)
    if done:
        store.write_once(f"{sample_dir(job, k)}/DONE",
                         json.dumps({"rows": len(rows), "shards": info}, sort_keys=True).encode())
    return info


def proef(args):
    """The proefrun on a local directory: three fake judges (MetricX parts 0 and 1, C2) serve while four writers
    stream two iterations, a val job and the anchor; one judge is killed mid-job and restarted; wait passes on every
    job, also with --full; a restart changes no score file (idempotent); a missing judge times out with the names of
    its shards; a stale heartbeat fails at once; STOP ends the judges with exit 0."""
    import shutil
    import signal
    import subprocess

    root = Path(args.dir)
    if root.exists():
        raise SystemExit(f"{root} exists: the proefrun wants a new directory")
    root.mkdir(parents=True)
    store = LocalStore(root)
    me = str(Path(__file__).resolve())
    logs = root.parent / f"{root.name}-logs"
    logs.mkdir(exist_ok=True)
    checks = []

    def log(m):
        print(f"{utc_now()} {m}", flush=True)

    def check(name, ok, detail=""):
        checks.append((name, bool(ok)))
        log(f"CHECK {name}: {'ok' if ok else 'FAILED'} {detail}")

    def start(role, worker, part=None, delay=0.05):
        cmd = [sys.executable, "-u", me, "serve", "--rl", str(root), "--role", role, "--worker", worker,
               "--scorer", "fake", "--fake-delay", str(delay), "--poll", "0.2", "--heartbeat-every", "1",
               "--writers", "iter=0,1,2,3", "--writers", "val=3", "--writers", "anchor=0,1,2,3"]
        if part is not None:
            cmd += ["--part", str(part)]
        f = open(logs / f"{role}-{worker}.log", "a")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)

    def run_wait(job, writers, extra=(), judges=("metricx:w0:0", "metricx:w2:1", "c2:w1"), timeout=60):
        cmd = [sys.executable, me, "wait", "--rl", str(root), "--job", job, "--writers", writers,
               "--timeout", str(timeout), "--poll", "0.2", *extra]
        for j in judges:
            cmd += ["--judge", j]
        r = subprocess.run(cmd, capture_output=True, text=True)
        log(f"WAIT {job}: exit {r.returncode} {r.stdout.strip()[-400:]}")
        return r

    procs = {"w0": start("metricx", "w0", 0), "w2": start("metricx", "w2", 1), "w1": start("c2", "w1")}
    time.sleep(1.5)
    sources = {k: [(f"src{k}-{s}", f"English source {k} {s}.") for s in range(12)] for k in range(4)}
    # iteration 1: the writers stream; C2 is killed after the first shards and restarted
    import pyarrow  # noqa: F401  (imported once here, not by four writer threads at once)

    threads = [threading.Thread(target=write_sample_shards,
                                args=(store, "iter/1", k, fake_samples(f"1/{k}", sources[k], 8), 16, 0.5),
                                kwargs={"log": log}) for k in range(4)]
    for t in threads:
        t.start()

    def first(prefix, limit=20.0):
        """the score files under prefix once there is one (polled for at most limit s)"""
        end = time.time() + limit
        while time.time() < end:
            found = [n for n in store.list(prefix) if n.endswith(".parquet")]
            if found:
                return found
            time.sleep(0.05)
        return []

    early = first("iter/1/scores/c2/w1/")
    dones = [k for k in range(4) if store.exists(f"{sample_dir('iter/1', k)}/DONE")]
    check("streaming: scores before the generation's DONE", early and not dones,
          f"{len(early)} score files, DONE of writers {dones}")
    os.killpg(procs["w1"].pid, signal.SIGKILL)
    procs["w1"].wait()
    log("KILLED c2:w1 mid-job")
    for t in threads:
        t.join()
    procs["w1"] = start("c2", "w1")
    r = run_wait("iter/1", "0,1,2,3", ("--full",))
    check("iter/1 scored after a judge restart (wait --full)", r.returncode == 0 and "SCORES_OK" in r.stdout)
    before = {n: store.read(n) for n in store.list("iter/1/scores/")}
    os.killpg(procs["w0"].pid, signal.SIGTERM)
    procs["w0"].wait()
    procs["w0"] = start("metricx", "w0", 0)
    time.sleep(2.0)
    after = {n: store.read(n) for n in store.list("iter/1/scores/")}
    check("idempotent: a restarted judge rewrote nothing", before == after, f"{len(after)} files")
    c2 = [n for n in store.list("iter/1/scores/c2/w1/") if n.endswith(".parquet")]
    over = sum(sum(1 for x in read_table(store.read(n)) if x["c2_overflow"]) for n in c2)
    check("c2_overflow counted, never truncated", over > 0, f"{over} rows over {C2_MAX_TOKENS} tokens")
    # iteration 2, the val job (w3 only) and the anchor
    for k in range(4):
        write_sample_shards(store, "iter/2", k, fake_samples(f"2/{k}", sources[k], 8), 24, log=log)
        write_sample_shards(store, "anchor", k, fake_samples(f"a/{k}", sources[k], 1), 8, log=log)
    write_sample_shards(store, "val/2", 3, fake_samples("v/2", sources[3], 1), 5, log=log)
    for job, writers in (("iter/2", "0,1,2,3"), ("anchor", "0,1,2,3"), ("val/2", "3")):
        r = run_wait(job, writers, ("--full",))
        check(f"{job} scored", r.returncode == 0 and "SCORES_OK" in r.stdout)
    # a judge that lives but does not get to the job: the timeout names its shards
    procs["w9"] = start("c2", "w9", delay=1000)
    time.sleep(1.5)
    write_sample_shards(store, "iter/3", 0, fake_samples("3/0", sources[0], 8), 32, log=log)
    r = run_wait("iter/3", "0", judges=("metricx:w0:0", "metricx:w2:1", "c2:w1", "c2:w9"), timeout=3)
    check("timeout of a judge that never scored, with the shard names",
          r.returncode == 1 and "c2:w9 iter/3/samples/w0/shard-0.parquet" in r.stdout
          and "s after the last DONE" in r.stdout)
    os.killpg(procs["w9"].pid, signal.SIGKILL)
    procs["w9"].wait()
    # a stale heartbeat fails before the timeout
    shutil.copy(root / heartbeat_name("c2", "w1"), logs / "c2-w1-heartbeat.json")
    os.killpg(procs["w1"].pid, signal.SIGKILL)
    procs["w1"].wait()
    hb = json.loads(store.read(heartbeat_name("c2", "w1")))
    hb["epoch"] -= HEARTBEAT_MAX + 60
    store.overwrite(heartbeat_name("c2", "w1"), json.dumps(hb).encode())
    write_sample_shards(store, "iter/4", 0, fake_samples("4/0", sources[0], 8), 32, log=log)
    t0 = time.time()
    r = run_wait("iter/4", "0", timeout=60)
    check("a dead judge (old heartbeat) fails at once", r.returncode == 1 and "heartbeat" in r.stdout
          and time.time() - t0 < 30)
    r = subprocess.run([sys.executable, me, "alive", "--rl", str(root), "--judge", "metricx:w0:0",
                        "--judge", "metricx:w2:1"], capture_output=True, text=True)
    check("alive: the two MetricX judges", r.returncode == 0, r.stdout.strip())
    store.write_once(STOP, b"proef\n")
    codes = {w: p.wait(timeout=30) for w, p in procs.items() if w not in ("w1", "w9")}
    check("STOP ends the judges with exit 0", all(c == 0 for c in codes.values()), str(codes))
    states = {w: json.loads(store.read(heartbeat_name(r_, w)))["state"] for r_, w in (("metricx", "w0"), ("metricx", "w2"))}
    check("their heartbeats say stopped", all(s == "stopped" for s in states.values()), str(states))
    failed = [n for n, ok in checks if not ok]
    if failed:
        print(f"JUDGE_PROEF_FAILED {failed}", flush=True)
        return 1
    print(f"JUDGE_PROEF_OK {len(checks)} checks; logs in {logs}", flush=True)
    return 0


def read_table(data):
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pylist()


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--rl", required=True, help="$RL: gs://... or a local directory")
    s.add_argument("--role", required=True, choices=ROLES)
    s.add_argument("--worker", required=True, help="w0, w1, w2 (the judge's worker: its directory and heartbeat)")
    s.add_argument("--part", type=int, choices=(0, 1), help="MetricX's half: parity of sha256(uid)")
    s.add_argument("--scorer", default="real", choices=("real", "fake"))
    s.add_argument("--root", help="MetricX: the download directory of metricx_qe.py")
    s.add_argument("--dumb-code", help="C2: a checkout of the DUMB code (as naturalness.py)")
    s.add_argument("--model", help="C2: the classifier's directory (params.pkl, config.json)")
    s.add_argument("--batch-size", type=int, default=8, help="C2: per device")
    s.add_argument("--writers", action="append", help="kind=k,...: when a job of that kind is complete")
    s.add_argument("--poll", type=float, default=5.0)
    s.add_argument("--heartbeat-every", type=float, default=HEARTBEAT_EVERY)
    s.add_argument("--max-idle", type=float, default=6 * 3600, help="stop after so many s without work (0: never)")
    s.add_argument("--fake-delay", type=float, default=0.0)
    a = sub.add_parser("alive")
    w = sub.add_parser("wait")
    for p in (a, w):
        p.add_argument("--rl", required=True)
        p.add_argument("--judge", action="append", required=True, help="role:w<m>[:part]")
        p.add_argument("--heartbeat-max", type=float, default=HEARTBEAT_MAX)
        p.add_argument("--busy-max", type=float, default=BUSY_MAX)
    w.add_argument("--job", required=True, help="iter/<i>, val/<i> or anchor")
    w.add_argument("--writers", required=True, help="the generating workers k: 0,1,2,3 (val: 3)")
    w.add_argument("--timeout", type=float, default=SCORE_TIMEOUT)
    w.add_argument("--poll", type=float, default=10.0)
    w.add_argument("--full", action="store_true", help="also compare the uids (reads every file: in the region)")
    p = sub.add_parser("proef")
    p.add_argument("--dir", required=True, help="a new local directory as the bucket")
    args = parser.parse_args(argv)
    if args.command == "serve" and args.role == "metricx" and args.part is None:
        parser.error("--role metricx needs --part 0 or 1")
    return {"serve": serve, "alive": alive, "wait": wait, "proef": proef}[args.command](args)


if __name__ == "__main__":
    import traceback

    code = 0
    try:
        code = main() or 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # as naturalness.py and metricx_qe.py: a normal exit can wait in the JAX runtime's shutdown
    os._exit(code)
