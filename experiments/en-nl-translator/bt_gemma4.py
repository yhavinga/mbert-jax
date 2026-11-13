#!/usr/bin/env python3
"""A new back-translation (nl->en) of all 271,710 Dutch passages of the five published back-translation datasets with
Gemma 4 31B-it instead of TranslateGemma 27B (Yeb, 13 Nov): TranslateGemma's English is wordier than a human
translation (median 0.956 Dutch words per English word, human ~1.06), and a translator trained on it learns to
compress. The pilot of 12 Nov (bt_natural.py, mbert-ops/translator-sft/bt-pilot-natural-english.md) found natural-v1
near the human length (1.019) but with added formatting (hard checks 4.9%).

    python bt_gemma4.py probeplan --data DIR --pilot probe-ids.json --tokenizer TOK --out DIR
    python bt_gemma4.py plan --data DIR --tokenizer TOK --out DIR [--model gemma4-31b-bt --max-source N]
    python bt_gemma4.py work --plan DIR --root gs://.../full --length-class small --prompt P --owner VM/h0 --run ID
    python bt_gemma4.py status --plan DIR --root gs://.../full
    python bt_gemma4.py select --plan DIR --shards DIR [--qe qe.parquet] --out DIR
    python bt_gemma4.py publish --plan DIR --root gs://.../full --data DIR --repo REPO --region R --owner VM
    python bt_gemma4.py card --repo REPO --prompt P --out README.md
    python bt_gemma4.py report --data DIR --out bt-gemma4.md

Step 1, the prompt (probeplan, work on the probe plan, select): ~300 passages, per source 40 uniform (stratum random,
from the train rows outside the pilot's sample), 7 long (385-511 TranslateGemma tokens), 7 from the lowest decile of
Dutch words per English word of TranslateGemma's English (ratio_low), and 6 of the pilot's natural-v1 translations
that failed `structure` (stratum structure); plus 20 context turns of the Handelingen (stratum context). Each prompt of
PROMPTS translates all of them. THE RULE (fixed on 13 Nov before any translation of the probe, Yeb's order), on the
stratum random only (200 passages: the other strata are chosen hard cases, not a sample of the data): a prompt is
eligible when its share of hard-check failures (pairs.hard_fails, or chatter) is at most 1.5%; of the eligible ones
the prompt whose median Dutch words per English word is closest to 1.06 (the median of R2's real pairs, Europarl and
Global Voices); where the distances to 1.06 differ by less than 0.01, the one with fewer hard-check failures. When no
prompt is eligible: the one with the fewest failures, reported as such. choose() is the rule.

Step 2, the full run (plan, work, publish): the passages of every source and split, and the Handelingen's context
turns, in units of SHARD_SIZE consecutive ids of one source, split and length class. Every process (4 chips, half a
v6e-8) takes units of one class at a time: it claims a unit in its region's bucket (an object created only if absent;
a claim older than the lease, or of an earlier run on the same VM, may be taken over), translates it and stores the
shard there. us-east1 goes through the units forwards, europe-west4 backwards; other-done.txt in each region's bucket
(written from co1, which lists both buckets) names the units the other region has done, so that no VM reads the
other region's bucket. publish (one process a region, under a lease) puts each shard, with the published dataset's
columns, the new English and its checks, as one parquet file into a private dataset on Hugging Face: the dataset is
the merge. Nothing is filtered out.
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

import bt_natural as bn  # noqa: E402

# name: (key of bt_natural.SOURCES, directory of that dataset on Hugging Face); one config of the new dataset each
PARTS = {
    "fineweb": ("fineweb", "data"),
    "handelingen": ("handelingen", "data"),
    "handelingen_context": ("handelingen", "context_turns"),
    "raad": ("raad", "data"),
    "dpc": ("dpc", "data"),
    "rechtspraak": ("rechtspraak", "data"),
}
SPLITS = ("train", "validation")
EXPECTED_PASSAGES = 271_710
MODEL = "gemma4-31b-bt"
MODEL_ID = "google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475"
PROMPTS = ("natural-v1", "natural-v1-fmt", "natural-v1-fmt-len")
SHORT = {"natural-v1": "v1", "natural-v1-fmt": "fmt", "natural-v1-fmt-len": "fmtlen"}
SHARD_SIZE = 512
PROBE_SEED = "bt-gemma4/probe/0"
PROBE_STRATA = (("random", 40), ("long", 7), ("ratio_low", 7))
PROBE_STRUCTURE = 6
PROBE_CONTEXT = 20
# The rule (module docstring)
TARGET_RATIO = 1.06
MAX_HARD = (3, 200)  # 1.5% as a fraction of integers: hard * 200 <= n * 3
TIE = 0.01
RULE_STRATUM = "random"
# R2's real pairs (Europarl + Global Voices), Dutch words per English word (data-r2-report.md; Yeb's order of 13 Nov)
HUMAN = {"median": 1.062, "p5": 0.889, "p95": 1.278}
LEASE_SECONDS = 20 * 60
PUBLISH_LEASE_SECONDS = 15 * 60
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def stable(key):
    return hashlib.sha1(key.encode()).hexdigest()


def now():
    return time.time()


def utc(t=None):
    return datetime.datetime.fromtimestamp(now() if t is None else t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------------- stores

class MemoryStore:
    """A bucket directory in memory, for the tests: key -> (bytes, generation)"""

    def __init__(self):
        self.objects, self.generation = {}, 0

    def _bump(self, key, data):
        self.generation += 1
        self.objects[key] = (data, self.generation)

    def list(self, prefix):
        return {k[len(prefix):] for k in self.objects if k.startswith(prefix)}

    def read(self, key):
        if key not in self.objects:
            return None
        data, gen = self.objects[key]
        return data.decode(), gen

    def create(self, key, text):
        if key in self.objects:
            return False
        self._bump(key, text.encode())
        return True

    def replace(self, key, text, generation):
        if key not in self.objects or self.objects[key][1] != generation:
            return False
        self._bump(key, text.encode())
        return True

    def exists(self, key):
        return key in self.objects

    def put(self, path, key):
        self._bump(key, Path(path).read_bytes())

    def get(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.objects[key][0])


class GcsStore:
    """A bucket directory (gs://bucket/prefix) through the Python client (translate.py: gsutil is a snap on v6e and
    refuses to start under a process with no_new_privs); create and replace use generation preconditions"""

    def __init__(self, root):
        import translate as tr

        self.client = tr.gcs_client()
        name, prefix = tr.split_gs(root.rstrip("/"))
        self.bucket, self.prefix = self.client.bucket(name), prefix.rstrip("/")

    def _name(self, key):
        return f"{self.prefix}/{key}"

    def list(self, prefix):
        full = self._name(prefix)
        return {b.name[len(full):] for b in self.client.list_blobs(self.bucket, prefix=full)}

    def read(self, key):
        """(text, generation), or None when absent or changed while being read (the caller tries again later)"""
        from google.api_core.exceptions import NotFound, PreconditionFailed

        blob = self.bucket.blob(self._name(key))
        try:
            blob.reload()
            return blob.download_as_text(if_generation_match=blob.generation), blob.generation
        except (NotFound, PreconditionFailed):
            return None

    def _write(self, key, text, generation):
        from google.api_core.exceptions import NotFound, PreconditionFailed

        try:
            self.bucket.blob(self._name(key)).upload_from_string(text, if_generation_match=generation)
            return True
        except (PreconditionFailed, NotFound):
            return False

    def create(self, key, text):
        return self._write(key, text, 0)

    def replace(self, key, text, generation):
        return self._write(key, text, generation)

    def exists(self, key):
        return self.bucket.blob(self._name(key)).exists()

    def put(self, path, key):
        from google.cloud.storage.retry import DEFAULT_RETRY

        self.bucket.blob(self._name(key)).upload_from_filename(str(path), retry=DEFAULT_RETRY)

    def get(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(self._name(key)).download_to_filename(str(path))


def claim(store, key, me, lease, t=None):
    """Claim key for me ({owner, run}): created if absent; taken over when the claim is older than lease seconds, or
    of another run of the same VM (a VM started again after a preemption keeps its name). Returns whether it is mine."""
    t = now() if t is None else t
    body = json.dumps({**me, "time": t})
    if store.create(key, body):
        return True
    found = store.read(key)
    if found is None:  # gone in between: the next pass tries again
        return False
    text, generation = found
    old = json.loads(text)
    mine = old.get("owner") == me["owner"] and old.get("run") == me["run"]  # a lease renewed by its holder
    same_vm = old.get("owner", "").split("/")[0] == me["owner"].split("/")[0] and old.get("run") != me["run"]
    if mine or same_vm or old.get("released") or t - old.get("time", 0) > lease:
        return store.replace(key, body, generation)
    return False


def release(store, key, me):
    """Give up a lease (claim treats it as free) without deleting the object"""
    found = store.read(key)
    if found and json.loads(found[0]).get("owner") == me["owner"]:
        store.replace(key, json.dumps({**me, "released": True, "time": now()}), found[1])


# ---------------------------------------------------------------------------------------------------- data

def part_files(data, part, split):
    key, folder = PARTS[part]
    return bn.source_files(data, key, folder, split)


def read_part(data, part, split, columns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa.concat_tables([pq.read_table(p, columns=columns) for p in part_files(data, part, split)])


def classes_of(model):
    import translate as tr

    return tr.MODELS[model]["length_classes"]


def write_segments(rows, path):
    """translate.py's segments (id, text, tokens), sorted by id"""
    rows = sorted(rows, key=lambda r: r["id"])
    bn.write_parquet([{"id": r["id"], "text": r["text"], "tokens": r["tokens"]} for r in rows], path)
    return rows


def plan_hash(units):
    h = hashlib.sha1()
    for u in units:
        h.update(f"{u['name']}|{u['first']}|{u['last']}|{u['n']}\n".encode())
    return h.hexdigest()


def make_units(segments_by_key, class_names, shard_size, part_order):
    """Units of shard_size consecutive segments of one (part, split, class), in the order class, part, split, k"""
    units = []
    for (part, split, cls), (path, rows) in segments_by_key.items():
        for k, start in enumerate(range(0, len(rows), shard_size)):
            chunk = rows[start:start + shard_size]
            units.append({"name": f"{part}/{split}-{cls}-{k:04d}", "part": part, "split": split, "class": cls,
                          "prompt": None, "segments": path, "start": start, "end": start + len(chunk),
                          "n": len(chunk), "first": chunk[0]["id"], "last": chunk[-1]["id"]})
    units.sort(key=lambda u: (class_names.index(u["class"]), part_order.index(u["part"]), SPLITS.index(u["split"]),
                              u["name"]))
    return units


def plan(args):
    """The full run's units: every passage of PARTS (train and validation), by its Gemma 4 tokens in the model's
    classes; a passage above --max-source (default the largest class) stops the plan"""
    classes = classes_of(args.model)
    names = list(classes)
    largest = classes[names[-1]]["max_source"]
    max_source = args.max_source or largest
    if max_source > largest:
        raise SystemExit(f"--max-source {max_source} is beyond the largest class of {args.model} ({largest})")
    count = bn.Counter(args.tokenizer)
    out = Path(args.out)
    segments_by_key, total = {}, 0
    for part in PARTS:
        for split in SPLITS:
            table = read_part(args.data, part, split, ["id", "nl"])
            ids, texts = table.column("id").to_pylist(), table.column("nl").to_pylist()
            tokens = count.many(texts)
            over = [i for i, n in zip(ids, tokens) if n > max_source]
            if over:
                raise SystemExit(f"{part}/{split}: {len(over)} passages above {max_source} tokens, e.g. {over[:3]}")
            per = collections.defaultdict(list)
            for i, text, n in zip(ids, texts, tokens):
                per[bn.class_of(n, classes)].append({"id": i, "text": text, "tokens": n})
            for cls, rows in per.items():
                rel = f"segments/{part}/{split}-{cls}.parquet"
                segments_by_key[(part, split, cls)] = (rel, write_segments(rows, out / rel))
            total += len(ids)
            print(f"PLAN {json.dumps({'part': part, 'split': split, 'passages': len(ids), 'classes': {c: len(r) for c, r in sorted(per.items())}})}", flush=True)
    if args.expect is not None and total != args.expect:
        raise SystemExit(f"{total} passages, expected {args.expect}")
    units = make_units(segments_by_key, names, args.shard_size, list(PARTS))
    info = {"model": args.model, "classes": classes, "max_source": max_source, "shard_size": args.shard_size,
            "passages": total, "units": units, "hash": plan_hash(units),
            "revisions": {k: v[1] for k, v in bn.SOURCES.items()}}
    (out / "units.json").write_text(json.dumps(info, indent=0))
    print(f"UNITS {json.dumps({'units': len(units), 'passages': total, 'hash': info['hash'], 'per_class': dict(collections.Counter(u['class'] for u in units))})}", flush=True)


def probe_draw(rows_by_source, context_rows, pilot, seed=PROBE_SEED):
    """The probe's passages: per source the strata of PROBE_STRATA from the train rows outside the pilot's sample, then
    PROBE_STRUCTURE of the pilot's structure failures of that source; PROBE_CONTEXT context turns"""
    exclude = set(pilot["sample_ids"])
    structure = set(pilot["structure_ids"])
    out, info = [], {}
    for source, rows in rows_by_source.items():
        fresh = [r for r in rows if r["id"] not in exclude]
        drawn, info[source] = bn.draw(fresh, source, strata=PROBE_STRATA, seed=seed)
        cases = sorted((r for r in rows if r["id"] in structure), key=lambda r: stable(f"{seed}/{r['id']}"))
        taken = [dict(r, source=source, stratum="structure") for r in cases[:PROBE_STRUCTURE]]
        if len(taken) < PROBE_STRUCTURE:
            info[source]["short"]["structure"] = [len(taken), PROBE_STRUCTURE]
        out += drawn + taken
    turns = sorted(context_rows, key=lambda r: stable(f"{seed}/{r['id']}"))[:PROBE_CONTEXT]
    out += [dict(r, source="handelingen", stratum="context") for r in turns]
    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("a probe passage is in two strata")
    return out, info


def probeplan(args):
    """The probe's passages (sample.parquet) and units: one per prompt and class, all passages of the class"""
    import translate as tr
    from transformers import AutoTokenizer

    pilot = json.loads(Path(args.pilot).read_text())
    out = Path(args.out)
    rows_by_source = {}
    for part, (key, folder) in PARTS.items():
        if folder != "data":
            continue
        table = read_part(args.data, part, "train", bn.COLUMNS)
        rows_by_source[key] = table.to_pylist()
    context = read_part(args.data, "handelingen_context", "train",
                        ["id", "nl", "en", "qe_en_nl", "hard_fails", "ended", "tokens_nl", "tokens_en"]).to_pylist()
    for r in context:
        r["length_class"] = None
    rows, info = probe_draw(rows_by_source, context, pilot)
    classes = classes_of(args.model)
    count = bn.Counter(args.tokenizer)
    for r, n in zip(rows, count.many(r["nl"] for r in rows)):
        r["tokens_g4"], r["class_g4"] = n, bn.class_of(n, classes)
    rows.sort(key=lambda r: r["id"])
    bn.write_parquet(rows, out / "sample.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    segs = [{"id": r["id"], "text": r["nl"], "tokens": r["tokens_g4"]} for r in rows]
    overhead = {}
    for name in PROMPTS:
        codec = tr.ModelPrompts(tokenizer, tr.with_prompt(tr.MODELS[args.model], name), "nl", "en")
        codec.check([s["text"] for s in segs[:3]])
        worst = 0
        for s in segs:
            n = len(codec.encode(s["text"]))
            limit = classes[bn.class_of(s["tokens"], classes)]["prefill_lengths"][-1]
            if n > limit:
                raise SystemExit(f"{s['id']}: prompt {name} has {n} tokens, more than its class's prefill {limit}")
            worst = max(worst, n - s["tokens"])
        overhead[name] = worst
    units = []
    for cls in classes:
        chosen = [s for s in segs if bn.class_of(s["tokens"], classes) == cls]
        if not chosen:
            continue
        rel = f"segments/probe-{cls}.parquet"
        write_segments(chosen, out / rel)
        for name in PROMPTS:
            units.append({"name": f"{SHORT[name]}/{cls}", "part": "probe", "split": "train", "class": cls,
                          "prompt": name, "segments": rel, "start": 0, "end": len(chosen), "n": len(chosen),
                          "first": chosen[0]["id"], "last": chosen[-1]["id"]})
    summary = {"rows": len(rows), "strata": dict(collections.Counter(r["stratum"] for r in rows)),
               "classes": dict(collections.Counter(r["class_g4"] for r in rows)), "prompt_overhead": overhead,
               "draw": info}
    (out / "units.json").write_text(json.dumps({"model": args.model, "classes": classes, "shard_size": None,
                                                "passages": len(rows), "units": units, "hash": plan_hash(units),
                                                "probe": summary}, indent=0))
    print(f"PROBEPLAN {json.dumps(summary)}", flush=True)


# ---------------------------------------------------------------------------------------------------- work

def load_plan(path):
    return json.loads((Path(path) / "units.json").read_text())


class Progress:
    """The units done in this region (shards/) and in the other (other-done.txt), read again after `every` seconds"""

    def __init__(self, store, every=30):
        self.store, self.every, self.read_at = store, every, None
        self.done, self.other = set(), set()

    def refresh(self, force=False):
        if force or self.read_at is None or now() - self.read_at > self.every:
            self.done = {k[:-len(".parquet")] for k in self.store.list("shards/") if k.endswith(".parquet")}
            found = self.store.read("other-done.txt")
            self.other = set(found[0].split()) if found else set()
            self.read_at = now()

    def finished(self, name):
        self.refresh()
        return name in self.done or name in self.other


def ordered(units, direction):
    return list(units) if direction == "forward" else list(reversed(units))


def work_units(units, store, me, translate, lease=LEASE_SECONDS, progress=None, log=print):
    """For each unit in order not done here or there: claim it, translate it (translate(unit) -> (rows, seconds),
    rows a dict of columns) and store its shard. Returns the names done."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    progress = progress or Progress(store)
    done = []
    for u in units:
        if progress.finished(u["name"]):
            continue
        if not claim(store, f"claims/{u['name']}", me, lease):
            continue
        if store.exists(f"shards/{u['name']}.parquet"):  # finished by another process since the last listing
            continue
        rows, seconds = translate(u)
        n = len(rows["id"])
        rows = dict(rows, unit=[u["name"]] * n, prompt=[u["prompt"]] * n, owner=[me["owner"]] * n,
                    region=[me.get("region")] * n, time=[utc()] * n)
        local = Path(me.get("work_dir", "/tmp")) / "shards" / f"{u['name']}.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows), local, compression="zstd")
        store.put(local, f"shards/{u['name']}.parquet")
        local.unlink()
        done.append(u["name"])
        n_in, n_out = sum(rows["tokens_in"]), sum(rows["tokens_out"])
        log(f"SHARD {json.dumps({'unit': u['name'], 'prompt': u['prompt'], 'segments': n, 'tokens_in': n_in, 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / max(seconds, 1e-9)), 'not_ended': rows['ended'].count(False)})}")
    return done


class Translator:
    """Gemma 4 on this process's 4 chips for one length class: the engine is made at the first unit"""

    def __init__(self, args, plan_info):
        import translate as tr

        self.tr, self.args, self.plan = tr, args, plan_info
        self.model = tr.MODELS[plan_info["model"]]
        self.shapes = dict(plan_info["classes"][args.length_class], slots=self.model["slots"][args.length_class])
        self.device = self.tokenizer = None
        self.codecs, self.tables = {}, {}

    def _start(self):
        import jax
        from transformers import AutoTokenizer

        tr, args = self.tr, self.args
        if (jax.device_count(), jax.process_count()) != (self.model["chips"], 1):
            raise SystemExit(f"needs one process with {self.model['chips']} chips, has {jax.process_count()} "
                             f"process(es) with {jax.device_count()} devices")
        ckpt = args.ckpt or tr.checkpoint_of(self.model, tr.vm_region())
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"MAXTEXT {json.dumps({'model': self.plan['model'], 'class': args.length_class, 'shapes': self.shapes, 'ckpt': ckpt, **tr.maxtext_source(), 'patches': tr.patch_maxtext()})}", flush=True)
        started = now()
        engine_args = argparse.Namespace(model=self.plan["model"], ckpt=ckpt, work_dir=args.work_dir,
                                         tokenizer=args.tokenizer)
        engine, params = tr.make_engine(engine_args, self.shapes, self.model["chips"])
        self.device = tr.EngineSlots(engine, params, self.shapes["prefill_lengths"])
        print(f"engine ready in {now() - started:.0f} s", flush=True)

    def codec(self, prompt):
        if prompt not in self.codecs:
            tr = self.tr
            self.codecs[prompt] = tr.ModelPrompts(self.tokenizer, tr.with_prompt(self.model, prompt), "nl", "en")
        return self.codecs[prompt]

    def segments(self, u):
        import pyarrow.parquet as pq

        if u["segments"] not in self.tables:
            self.tables[u["segments"]] = pq.read_table(Path(self.args.plan) / u["segments"]).to_pylist()
        rows = self.tables[u["segments"]][u["start"]:u["end"]]
        if not rows or rows[0]["id"] != u["first"] or rows[-1]["id"] != u["last"] or len(rows) != u["n"]:
            raise SystemExit(f"{u['name']}: the segments file does not hold the plan's unit")
        return rows

    def __call__(self, u):
        if self.device is None:
            self._start()
        rows = self.segments(u)
        prompt = u["prompt"] or self.args.prompt
        codec = self.codec(prompt)
        texts = [r["text"] for r in rows]
        codec.check(texts[:2] + texts[-1:])
        out, seconds = self.tr.translate_shard(self.device, codec, self.tokenizer, [r["id"] for r in rows], texts,
                                               self.shapes, self.args.sync_every, None, self.model["eos_ids"])
        u["prompt"] = prompt
        return out, seconds


def work(args):
    plan_info = load_plan(args.plan)
    if args.length_class not in plan_info["classes"]:
        raise SystemExit(f"--length-class {args.length_class} is not a class of the plan")
    units = ordered([dict(u) for u in plan_info["units"] if u["class"] == args.length_class], args.direction)
    if not args.prompt and any(u["prompt"] is None for u in units):
        raise SystemExit("--prompt is needed for the units of the full plan")
    store = GcsStore(args.root)
    me = {"owner": args.owner, "run": args.run, "region": args.region, "work_dir": args.work_dir}
    translator = Translator(args, plan_info)
    started = now()
    done = work_units(units, store, me, translator, args.lease, log=lambda s: print(s, flush=True))
    print(f"CLASS {json.dumps({'class': args.length_class, 'units': len(units), 'done_here': len(done), 'engine': translator.device is not None, 'seconds': round(now() - started)})}", flush=True)


def region_status(plan_info, progress):
    progress.refresh(force=True)
    per = collections.defaultdict(lambda: {"units": 0, "here": 0, "other": 0, "pending": 0})
    for u in plan_info["units"]:
        c = per[u["class"]]
        c["units"] += 1
        if u["name"] in progress.done:
            c["here"] += 1
        elif u["name"] in progress.other:
            c["other"] += 1
        else:
            c["pending"] += 1
    pending = sum(c["pending"] for c in per.values())
    return {"classes": dict(per), "pending": pending, "complete": pending == 0}


def status(args):
    s = region_status(load_plan(args.plan), Progress(GcsStore(args.root)))
    print(json.dumps(s["classes"]))
    print("COMPLETE" if s["complete"] else f"PENDING {s['pending']}", flush=True)


# ---------------------------------------------------------------------------------------------------- measures

def g4_columns(nl, en, part, ended):
    """The new English's checks as columns (gemma4_*): pairs.measure and hard_fails, chatter, word and character
    ratios, R2's length bound of the source"""
    m = bn.measure(nl, en, PARTS[part][0], ended)
    return {"gemma4_hard_fails": m["hard_fails"] + (["chatter"] if m["chatter"] else []),
            "gemma4_structure": m["structure"], "gemma4_chatter": m["chatter"],
            "gemma4_decode_loop": m["decode_loop"], "gemma4_dutch_left": m["dutch_left"],
            "gemma4_len_ratio": m["len_ratio"], "gemma4_numbers_missing": m["numbers_missing"],
            "gemma4_word_ratio": m["word_ratio"], "gemma4_char_ratio": m["char_ratio"],
            "gemma4_r2_length_out": m["r2_length_out"]}


def g4_schema():
    import pyarrow as pa

    strings = pa.list_(pa.string())
    return [("en_gemma4", pa.string()), ("gemma4_ended", pa.bool_()), ("gemma4_tokens_nl", pa.int32()),
            ("gemma4_tokens_out", pa.int32()), ("gemma4_prompt", pa.string()), ("gemma4_prompt_text", pa.string()),
            ("gemma4_model", pa.string()), ("gemma4_hard_fails", strings), ("gemma4_structure", strings),
            ("gemma4_chatter", pa.bool_()), ("gemma4_decode_loop", pa.bool_()), ("gemma4_dutch_left", pa.bool_()),
            ("gemma4_len_ratio", pa.float64()), ("gemma4_numbers_missing", strings),
            ("gemma4_word_ratio", pa.float64()), ("tg_word_ratio", pa.float64()), ("gemma4_char_ratio", pa.float64()),
            ("gemma4_r2_length_out", pa.bool_()), ("gemma4_region", pa.string()), ("gemma4_vm", pa.string()),
            ("gemma4_time", pa.string())]


def prompt_text(prompt):
    import translate as tr

    return tr.user_text(tr.PROMPTS[prompt], "{text}", "nl", "en")


def published_rows(unit, shard, source_rows, segments_tokens):
    """The rows of one unit for the dataset: the published dataset's columns of each passage, then the new English and
    its checks; in the order of the shard"""
    out = []
    texts = {}
    for i, en, ended, n_out, prompt, owner, region, t in zip(
            shard["id"], shard["text_en"], shard["ended"], shard["tokens_out"], shard["prompt"], shard["owner"],
            shard["region"], shard["time"]):
        src = source_rows.get(i)
        if src is None:
            raise SystemExit(f"{unit['name']}: {i} is not a passage of {unit['part']}/{unit['split']}")
        if prompt not in texts:
            texts[prompt] = prompt_text(prompt)
        row = dict(src)
        row.update({"en_gemma4": en, "gemma4_ended": ended, "gemma4_tokens_nl": segments_tokens[i],
                    "gemma4_tokens_out": n_out, "gemma4_prompt": prompt, "gemma4_prompt_text": texts[prompt],
                    "gemma4_model": MODEL_ID})
        row.update(g4_columns(src["nl"], en, unit["part"], ended))
        row["tg_word_ratio"] = round(bn.word_ratio(src["nl"], src["en"]), 4)
        row.update({"gemma4_region": region, "gemma4_vm": owner, "gemma4_time": t})
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------- select

def choose(stats):
    """THE RULE (module docstring). stats: {prompt: {"n", "hard", "median"}} of the stratum random, in the order of
    PROMPTS. Returns (prompt, reason, table)."""
    table = []
    for prompt, s in stats.items():
        table.append({"prompt": prompt, "n": s["n"], "hard": s["hard"], "share": s["hard"] / s["n"],
                      "median": s["median"], "distance": abs(s["median"] - TARGET_RATIO),
                      "eligible": s["hard"] * MAX_HARD[1] <= s["n"] * MAX_HARD[0]})
    eligible = [t for t in table if t["eligible"]]
    if eligible:
        best = min(t["distance"] for t in eligible)
        close = [t for t in eligible if t["distance"] - best < TIE]
        pick = min(close, key=lambda t: (t["hard"] / t["n"], t["distance"]))
        reason = ("rule: eligible (hard <= 1.5%), closest to 1.06" if len(close) == 1 else
                  f"rule: eligible, within {TIE} of the closest to 1.06: {', '.join(t['prompt'] for t in close)}; "
                  "the fewest hard-check failures")
    else:
        pick = min(table, key=lambda t: (t["hard"] / t["n"], t["distance"]))
        reason = "no prompt has hard <= 1.5%: the fewest hard-check failures (reported as such)"
    return pick["prompt"], reason, table


def setup_stats(rows, key):
    ratios = [r[f"word_ratio_{key}"] for r in rows]
    hard = sum(bool(r[f"hard_{key}"]) for r in rows)
    cats = collections.Counter(c for r in rows for c in (r[f"structure_{key}"] or []))
    checks = collections.Counter(c for r in rows for c in r[f"hard_{key}"])
    return {"n": len(rows), "hard": hard, "median": statistics.median(ratios) if ratios else None,
            "quantiles": {f"p{round(q * 100)}": bn.quantile(ratios, q) for q in QUANTILES} if ratios else {},
            "under_0_9": sum(x < 0.9 for x in ratios) / len(ratios) if ratios else None,
            "structure_categories": dict(cats), "checks": dict(checks)}


def select(args):
    """The probe's measures per prompt and the rule's choice: compare.parquet, selection.json, selection.md"""
    plan_info = load_plan(args.plan)
    sample = bn.read_rows(Path(args.plan) / "sample.parquet")
    shards = {}
    for path in sorted(Path(args.shards).rglob("*.parquet")):
        for r in bn.read_rows(path):
            key = SHORT[r["prompt"]]
            if r["id"] in shards.setdefault(key, {}):
                raise SystemExit(f"{r['id']} twice for {r['prompt']}")
            shards[key][r["id"]] = r
    qe = {}
    if args.qe:
        qe = {r["id"]: r for r in bn.read_rows(args.qe)}
    keys = ["tg"] + [SHORT[p] for p in PROMPTS]
    rows = []
    for p in sample:
        part = "handelingen_context" if p["stratum"] == "context" else p["source"]
        row = {k: p[k] for k in ("id", "source", "stratum", "class_g4", "tokens_g4", "nl")}
        for key in keys:
            if key == "tg":
                en, ended = p["en"], p["ended"]
            else:
                t = shards.get(key, {}).get(p["id"])
                if t is None:
                    raise SystemExit(f"{key}: no translation of {p['id']}")
                en, ended = t["text_en"], t["ended"]
            m = g4_columns(p["nl"], en, part, ended)
            row[f"en_{key}"] = en
            row[f"hard_{key}"] = m["gemma4_hard_fails"]
            row[f"structure_{key}"] = m["gemma4_structure"]
            row[f"word_ratio_{key}"] = m["gemma4_word_ratio"]
            for name in (f"qe_nl_en_{key}", f"qe_en_{key}_nl"):
                if name in qe.get(p["id"], {}):
                    row[name] = qe[p["id"]][name]
        rows.append(row)
    out = Path(args.out)
    bn.write_parquet(rows, out / "compare.parquet")
    by_stratum = collections.defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    stats = {s: {k: setup_stats(rs, k) for k in keys} for s, rs in sorted(by_stratum.items())}
    rule_rows = by_stratum[RULE_STRATUM]
    prompt, reason, table = choose({p: {"n": stats[RULE_STRATUM][SHORT[p]]["n"],
                                        "hard": stats[RULE_STRATUM][SHORT[p]]["hard"],
                                        "median": stats[RULE_STRATUM][SHORT[p]]["median"]} for p in PROMPTS})
    per_source = {src: {k: setup_stats([r for r in rule_rows if r["source"] == src], k) for k in keys}
                  for src in sorted({r["source"] for r in rule_rows})}
    qe_stats = {}
    if qe:
        for s, rs in by_stratum.items():
            qe_stats[s] = {k: {d: statistics.fmean(r[c] for r in rs) for d, c in
                               (("nl_en", f"qe_nl_en_{k}"), ("en_nl", f"qe_en_{k}_nl")) if all(c in r for r in rs)}
                           for k in keys}
    result = {"created": utc(), "plan_hash": plan_info["hash"], "chosen": prompt, "reason": reason, "rule": table,
              "stats": stats, "per_source_random": per_source, "qe": qe_stats,
              "prompts": {p: prompt_text(p) for p in PROMPTS}}
    (out / "selection.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / "selection.md").write_text(selection_md(result, rows), encoding="utf-8")
    print(f"CHOSEN {json.dumps({'prompt': prompt, 'reason': reason})}", flush=True)


def selection_md(result, rows):
    num, pct = bn.nl_num, bn.pct
    names = {"tg": "TranslateGemma 27B (gepubliceerd)", "v1": "Gemma 4 natural-v1", "fmt": "Gemma 4 natural-v1-fmt",
             "fmtlen": "Gemma 4 natural-v1-fmt-len"}
    keys = list(names)
    lines = [f"# Promptkeuze voor de nieuwe BT met Gemma 4 31B-it (gegenereerd {result['created']})", "",
             "## Keuzeregel (vastgelegd vóór de uitslag)", "",
             "Alleen op het stratum `random` (200 passages, 40 per bron, uit de trainrijen buiten de pilotsteekproef). "
             "Een prompt komt in aanmerking als zijn aandeel harde-controlefouten (`pairs.hard_fails` of `chatter`) "
             "hoogstens 1,5% is (3 van 200). Daarvan de prompt met de mediaan NL/EN-woorden het dichtst bij 1,06; "
             "liggen de afstanden tot 1,06 minder dan 0,01 uit elkaar, dan de prompt met minder harde-controlefouten. "
             "Haalt geen prompt 1,5%: de prompt met de minste fouten.", "",
             "| prompt | n | hard | aandeel | mediaan NL/EN | afstand tot 1,06 | in aanmerking |", "|---|---:|---:|---:|---:|---:|---|"]
    for t in result["rule"]:
        lines.append(f"| `{t['prompt']}` | {t['n']} | {t['hard']} | {pct(t['share'])} | {num(t['median'], 3)} | "
                     f"{num(t['distance'], 3)} | {'ja' if t['eligible'] else 'nee'} |")
    lines += ["", f"**Gekozen: `{result['chosen']}`** ({result['reason']}).", "",
              "## Per stratum", "",
              "| stratum | opzet | n | NL/EN-woorden p5 / p25 / mediaan / p75 / p95 | < 0,9 | hard | structuur-categorieën |",
              "|---|---|---:|---|---:|---:|---|"]
    for s, per in result["stats"].items():
        for k in keys:
            st = per[k]
            q = st["quantiles"]
            lines.append(f"| `{s}` | {names[k]} | {st['n']} | {num(q['p5'], 3)} / {num(q['p25'], 3)} / "
                         f"{num(st['median'], 3)} / {num(q['p75'], 3)} / {num(q['p95'], 3)} | {pct(st['under_0_9'])} | "
                         f"{st['hard']} | {', '.join(f'{c} {n}' for c, n in sorted(st['structure_categories'].items(), key=lambda x: -x[1])) or '–'} |")
    lines += ["", "Menselijke doelverdeling (R2's echte paren, Europarl + Global Voices): mediaan 1,062, p5 0,889, p95 1,278.",
              "", "## Stratum `random` per bron", "", "| bron | opzet | mediaan NL/EN | hard |", "|---|---|---:|---:|"]
    for src, per in result["per_source_random"].items():
        for k in keys:
            lines.append(f"| {src} | {names[k]} | {num(per[k]['median'], 3)} | {per[k]['hard']} |")
    if result["qe"]:
        lines += ["", "## MetricX-24-QE (alleen ter rapportage)", "",
                  "| stratum | opzet | QE nl→en gem. | QE en→nl gem. |", "|---|---|---:|---:|"]
        for s, per in result["qe"].items():
            for k in keys:
                lines.append(f"| `{s}` | {names[k]} | {num(per[k].get('nl_en'))} | {num(per[k].get('en_nl'))} |")
    lines += ["", "## Prompts", ""]
    for p, text in result["prompts"].items():
        lines += [f"`{p}`:", "", "```text", text, "```", ""]
    lines += ["## Gevallen met toegevoegde opmaak in de pilot (stratum `structure`)", ""]
    for r in [r for r in rows if r["stratum"] == "structure"][:6]:
        lines += [f"### `{r['id']}`", "", f"- natural-v1: {r['hard_v1'] or '–'}; fmt: {r['hard_fmt'] or '–'}; "
                  f"fmt-len: {r['hard_fmtlen'] or '–'}", "", "```text", "NL: " + r["nl"][:600], "",
                  "fmt: " + (r["en_fmt"] or "")[:600], "```", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- publish

def hf_path(unit_name):
    part, rest = unit_name.split("/", 1)
    return f"{part}/{rest}.parquet"


class SourceRows:
    """The published dataset's rows of a (part, split) by id, read once from the local copy of Hugging Face"""

    def __init__(self, data):
        self.data, self.cache = data, {}

    def __call__(self, part, split):
        if (part, split) not in self.cache:
            table = read_part(self.data, part, split)
            self.cache[(part, split)] = (table.schema, {r["id"]: r for r in table.to_pylist()})
        return self.cache[(part, split)]


def unit_file(unit, shard_path, sources, tokens_of, out_path):
    """The parquet file of one unit for the dataset, with the published schema plus g4_schema()"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema, by_id = sources(unit["part"], unit["split"])
    shard = pq.read_table(shard_path).to_pydict()
    rows = published_rows(unit, shard, by_id, tokens_of(unit))
    full = pa.schema(list(schema) + [pa.field(n, t) for n, t in g4_schema()])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=full), out_path, compression="zstd")
    return len(rows)


def publish(args):
    """One process a region (a lease on publish.lock): each shard of this region not yet on Hugging Face, as one
    parquet file, in commits of at most --batch files every --interval seconds; ends once the region is complete and
    all its shards are on Hugging Face (writes region-complete-<region>.json)"""
    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    plan_info = load_plan(args.plan)
    units = {u["name"]: u for u in plan_info["units"]}
    store = GcsStore(args.root)
    progress = Progress(store)
    sources = SourceRows(args.data)
    seg_cache = {}

    def tokens_of(unit):
        if unit["segments"] not in seg_cache:
            t = pq.read_table(Path(args.plan) / unit["segments"], columns=["id", "tokens"]).to_pydict()
            seg_cache[unit["segments"]] = dict(zip(t["id"], t["tokens"]))
        return seg_cache[unit["segments"]]

    me = {"owner": args.owner, "run": args.run}
    work_dir = Path(args.work_dir)
    complete_key = f"region-complete-{args.region}.json"
    while True:
        if store.exists(complete_key) and region_status(plan_info, progress)["complete"]:
            print(f"{utc()} {complete_key} is there and the region is complete: nothing to publish", flush=True)
            return
        if not claim(store, "publish.lock", me, PUBLISH_LEASE_SECONDS):
            print(f"{utc()} publish.lock held by another process", flush=True)
            if args.once:
                return
            time.sleep(args.interval)
            continue
        state = region_status(plan_info, progress)
        on_hf = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")}
        new = [n for n in sorted(progress.done) if n in units and hf_path(n) not in on_hf]
        for k in range(0, len(new), args.batch):
            ops, rows = [], 0
            for name in new[k:k + args.batch]:
                shard = work_dir / "pub-shards" / f"{name}.parquet"
                store.get(f"shards/{name}.parquet", shard)
                out = work_dir / "pub-files" / hf_path(name)
                rows += unit_file(units[name], shard, sources, tokens_of, out)
                ops.append(CommitOperationAdd(path_in_repo=hf_path(name), path_or_fileobj=str(out)))
            api.create_commit(args.repo, repo_type="dataset", operations=ops,
                              commit_message=f"{len(ops)} shards from {args.region} ({rows} rows)")
            size = sum(Path(op.path_or_fileobj).stat().st_size for op in ops)
            for op in ops:
                Path(op.path_or_fileobj).unlink()
            print(f"PUBLISHED {json.dumps({'files': len(ops), 'rows': rows, 'bytes': size, 'region': args.region, 'time': utc()})}", flush=True)
            if not claim(store, "publish.lock", me, PUBLISH_LEASE_SECONDS):
                print("publish.lock lost", flush=True)
                break
        if state["complete"] and not new:
            on_hf = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")}
            missing_here = [n for n in progress.done if n in units and hf_path(n) not in on_hf]
            if not missing_here:
                all_on_hf = sum(hf_path(n) in on_hf for n in units)
                body = json.dumps({"region": args.region, "time": utc(), "units_here": len(progress.done),
                                   "units_on_hf": all_on_hf, "units": len(units)})
                found = store.read(complete_key)
                if found is None:
                    store.create(complete_key, body)
                else:
                    store.replace(complete_key, body, found[1])
                release(store, "publish.lock", me)
                print(f"REGION COMPLETE {body}", flush=True)
                return
        if args.once:
            return
        time.sleep(args.interval)


# ---------------------------------------------------------------------------------------------------- card, report

def card(args):
    import yaml

    configs = [{"config_name": part, "data_files": [{"split": s, "path": f"{part}/{s}-*"} for s in SPLITS]}
               for part in PARTS]
    meta = {"license": "other", "language": ["nl", "en"], "pretty_name": "Dutch passages back-translated to English by "
            "Gemma 4 31B-it", "task_categories": ["translation"], "tags": ["back-translation", "synthetic"],
            "configs": configs}
    text = prompt_text(args.prompt)
    body = f"""# Dutch passages back-translated to English by Gemma 4 31B-it (private, work in progress)

The 271,710 Dutch passages and context turns of the five back-translation datasets of the collection
[Dutch back-translation (nl→en)](https://huggingface.co/collections/yhavinga/dutch-back-translation-nlen-6af45e2ba548b444c6362255)
(revisions of 11 Nov 2026), translated again from Dutch into English with `{MODEL_ID}` instead of TranslateGemma 27B,
so that the synthetic English is about as long as a human translation. One config per source (`handelingen_context`:
the Handelingen's context turns), splits as in the source datasets. Every row keeps all columns of its source dataset
(TranslateGemma's English is `en`) and adds:

| column | meaning |
|---|---|
| `en_gemma4` | Gemma 4 31B-it's English translation of `nl` (greedy) |
| `gemma4_ended` | the translation stopped by itself |
| `gemma4_tokens_nl`, `gemma4_tokens_out` | Gemma 4 tokens of `nl` and of the output (with the stop token) |
| `gemma4_prompt`, `gemma4_prompt_text` | the prompt's name and its text (`{{text}}` stands for `nl`) |
| `gemma4_model` | model and revision |
| `gemma4_hard_fails` | the hard checks the pair (`nl`, `en_gemma4`) fails: `not_ended`, `decode_loop`, `dutch_left`, `len_ratio`, `structure`, `newline_doubled` (as `hard_fails` of the source datasets), and `chatter` (an introduction or a note around the translation) |
| `gemma4_structure`, `gemma4_chatter`, `gemma4_decode_loop`, `gemma4_dutch_left`, `gemma4_len_ratio`, `gemma4_numbers_missing` | the measures behind the checks, as in the source datasets |
| `gemma4_word_ratio`, `tg_word_ratio` | Dutch words per English word (split on whitespace) of `en_gemma4` and of `en` |
| `gemma4_char_ratio`, `gemma4_r2_length_out` | Dutch characters over English characters; outside the length bound of our R2 data (factor 4/3 around the source's median) |
| `gemma4_region`, `gemma4_vm`, `gemma4_time` | where and when the translation was made |

Nothing was filtered out. The prompt, in Gemma 4's chat template with an empty thought channel:

```text
{text}
```

Code: `experiments/en-nl-translator/bt_gemma4.py` of mbert-jax (branch `bt-gemma4`). The license of each config is that
of its source dataset; the English was made with Gemma 4, under the Gemma Terms of Use. Before this dataset is made
public its card must be completed (license per config, selection, limitations).

## Acknowledgements

The translations and scores in this dataset were produced on Cloud TPUs provided by Google's
[TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
"""
    Path(args.out).write_text("---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n\n" + body,
                              encoding="utf-8")


def ratio_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), **{f"p{round(q * 100)}": bn.quantile(values, q) for q in QUANTILES},
            "under_0_9": sum(v < 0.9 for v in values) / len(values),
            "within_human_p5_p95": sum(HUMAN["p5"] <= v <= HUMAN["p95"] for v in values) / len(values)}


def report_stats(rows):
    hard = collections.Counter(c for r in rows for c in r["gemma4_hard_fails"])
    cats = collections.Counter(c for r in rows for c in (r["gemma4_structure"] or []))
    tg_hard = collections.Counter(c for r in rows for c in (r["hard_fails"] or []))
    return {"n": len(rows), "g4": ratio_stats([r["gemma4_word_ratio"] for r in rows]),
            "tg": ratio_stats([r["tg_word_ratio"] for r in rows]),
            "g4_hard": sum(bool(r["gemma4_hard_fails"]) for r in rows) / len(rows),
            "tg_hard": sum(bool(r["hard_fails"]) for r in rows) / len(rows),
            "g4_checks": dict(hard), "g4_structure": dict(cats), "tg_checks": dict(tg_hard),
            "g4_not_ended": sum(not r["gemma4_ended"] for r in rows),
            "g4_r2_length_out": sum(r["gemma4_r2_length_out"] for r in rows) / len(rows)}


def report(args):
    """bt-gemma4.md and .json from the dataset's files (a local copy of the Hugging Face dataset)"""
    import pyarrow.parquet as pq

    cols = ["id", "nl", "en", "en_gemma4", "hard_fails", "gemma4_hard_fails", "gemma4_structure", "gemma4_ended",
            "gemma4_word_ratio", "tg_word_ratio", "gemma4_r2_length_out", "gemma4_prompt", "gemma4_region"]
    by_part, files, size = {}, 0, 0
    for part in PARTS:
        rows = []
        for path in sorted((Path(args.data) / part).glob("*.parquet")):
            rows += pq.read_table(path, columns=cols).to_pylist()
            files += 1
            size += path.stat().st_size
        by_part[part] = rows
    total = sum(len(r) for r in by_part.values())
    ids = [r["id"] for rs in by_part.values() for r in rs]
    stats = {part: report_stats(rows) for part, rows in by_part.items() if rows}
    stats["_all"] = report_stats([r for rs in by_part.values() for r in rs])
    result = {"created": utc(), "files": files, "bytes": size, "rows": total, "unique_ids": len(set(ids)),
              "expected": EXPECTED_PASSAGES, "human": HUMAN, "stats": stats,
              "prompts": dict(collections.Counter(r["gemma4_prompt"] for rs in by_part.values() for r in rs)),
              "regions": dict(collections.Counter(r["gemma4_region"] for rs in by_part.values() for r in rs))}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    Path(args.out).write_text(report_md(result, by_part), encoding="utf-8")
    print(f"REPORT {json.dumps({'rows': total, 'unique': len(set(ids)), 'files': files})}", flush=True)


def report_md(result, by_part):
    num, pct = bn.nl_num, bn.pct
    lines = [f"# Nieuwe BT nl→en met Gemma 4 31B-it: uitkomst (gegenereerd {result['created']})", "",
             f"Rijen {result['rows']:,} in {result['files']} parquet-bestanden ({result['bytes'] / 1e6:,.1f} MB, de totale "
             f"upload naar Hugging Face), unieke id's {result['unique_ids']:,} (verwacht {result['expected']:,}). "
             f"Prompts: {result['prompts']}. Regio's: {result['regions']}.", "",
             "## NL-woorden per EN-woord per bron", "",
             "| bron | opzet | n | p5 | p10 | p25 | mediaan | p75 | p90 | p95 | < 0,9 | binnen p5–p95 van de mens |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for part, st in result["stats"].items():
        for key, name in (("tg", "TranslateGemma 27B"), ("g4", "Gemma 4 31B")):
            q = st[key]
            lines.append(f"| {part} | {name} | {q['n']:,} | " + " | ".join(num(q[p], 3) for p in
                         ("p5", "p10", "p25", "p50", "p75", "p90", "p95")) +
                         f" | {pct(q['under_0_9'])} | {pct(q['within_human_p5_p95'])} |")
    h = result["human"]
    lines += ["", f"Menselijke doelverdeling (R2's echte paren, Europarl + Global Voices): mediaan {num(h['median'], 3)}, "
              f"p5 {num(h['p5'], 3)}, p95 {num(h['p95'], 3)}.", "", "## Harde controles per bron", "",
              "| bron | hard TG | hard Gemma 4 | Gemma 4 per controle | structuur-categorieën Gemma 4 | niet af | buiten R2-lengte |",
              "|---|---:|---:|---|---|---:|---:|"]
    for part, st in result["stats"].items():
        lines.append(f"| {part} | {pct(st['tg_hard'])} | {pct(st['g4_hard'])} | "
                     f"{', '.join(f'{c} {n}' for c, n in sorted(st['g4_checks'].items(), key=lambda x: -x[1])) or '–'} | "
                     f"{', '.join(f'{c} {n}' for c, n in sorted(st['g4_structure'].items(), key=lambda x: -x[1])) or '–'} | "
                     f"{st['g4_not_ended']} | {pct(st['g4_r2_length_out'])} |")
    lines += ["", "## Voorbeelden", ""]
    for part, rows in by_part.items():
        if not rows:
            continue
        pick = sorted(rows, key=lambda r: stable(f"report/{r['id']}"))
        shown = [r for r in pick if len(r["nl"].split()) <= 120][:1]
        shown += [r for r in pick if r["gemma4_hard_fails"] and len(r["nl"].split()) <= 120][:1]
        for r in shown:
            lines += [f"### {part} `{r['id']}` (NL/EN TG {num(r['tg_word_ratio'])}, Gemma 4 {num(r['gemma4_word_ratio'])}; "
                      f"hard Gemma 4: {r['gemma4_hard_fails'] or '–'})", "",
                      "| Nederlands | TranslateGemma 27B | Gemma 4 31B |", "|---|---|---|",
                      "| " + " | ".join(x.replace("\n", "<br>").replace("|", "\\|") for x in (r["nl"], r["en"], r["en_gemma4"])) + " |", ""]
    return "\n".join(lines) + "\n"


def final(args):
    """Once every unit of the plan is a file on Hugging Face (and no other file is): the report, once a region (a
    lease on report.lock), from a copy of the dataset on this VM, to <root>/report/"""
    from huggingface_hub import HfApi, snapshot_download

    plan_info = load_plan(args.plan)
    expected = {hf_path(u["name"]) for u in plan_info["units"]}
    on_hf = {f for f in HfApi().list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")}
    if on_hf - expected:
        raise SystemExit(f"{len(on_hf - expected)} files on Hugging Face that are no unit of the plan, e.g. "
                         f"{sorted(on_hf - expected)[:3]}")
    if expected - on_hf:
        print(f"NOT COMPLETE: {len(expected - on_hf)} of {len(expected)} units not yet on Hugging Face", flush=True)
        return
    store = GcsStore(args.root)
    if store.exists("report/bt-gemma4.md"):
        print("the report is there", flush=True)
        return
    me = {"owner": args.owner, "run": args.run}
    if not claim(store, "report.lock", me, 3600):
        print("report.lock held by another VM", flush=True)
        return
    work_dir = Path(args.work_dir)
    local = snapshot_download(args.repo, repo_type="dataset", local_dir=str(work_dir / "dataset"),
                              allow_patterns=["*.parquet", "README.md"])
    out = work_dir / "report" / "bt-gemma4.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    report(argparse.Namespace(data=local, out=str(out)))
    for path in (out, out.with_suffix(".json")):
        store.put(path, f"report/{path.name}")
    print(f"FINAL report in {args.root}/report/", flush=True)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probeplan")
    p.add_argument("--data", required=True)
    p.add_argument("--pilot", required=True, help="probe-ids.json: sample_ids and structure_ids of the pilot")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-source", type=int, default=None,
                   help="The largest passage in Gemma 4 tokens (default: the model's largest class)")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--expect", type=int, default=EXPECTED_PASSAGES, help="Passages expected (0: no check)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("work")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True, help="gs:// directory of this region: shards/, claims/, other-done.txt")
    p.add_argument("--length-class", required=True)
    p.add_argument("--prompt", default=None, help="The prompt of the units without one (the full plan)")
    p.add_argument("--direction", choices=("forward", "backward"), default="forward")
    p.add_argument("--owner", default=f"{socket.gethostname()}/h0")
    p.add_argument("--run", default=str(int(now())))
    p.add_argument("--region", default=None)
    p.add_argument("--lease", type=int, default=LEASE_SECONDS)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sync-every", type=int, default=8)
    p = sub.add_parser("status")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p = sub.add_parser("select")
    p.add_argument("--plan", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--qe", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--owner", default=f"{socket.gethostname()}/publish")
    p.add_argument("--run", default=str(int(now())))
    p.add_argument("--work-dir", required=True)
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--batch", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p = sub.add_parser("final")
    p.add_argument("--plan", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--owner", default=f"{socket.gethostname()}/final")
    p.add_argument("--run", default=str(int(now())))
    p.add_argument("--work-dir", required=True)
    p = sub.add_parser("card")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--data", required=True, help="A local copy of the dataset (one directory per config)")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if getattr(args, "expect", None) == 0:
        args.expect = None
    {"probeplan": probeplan, "plan": plan, "work": work, "status": status, "select": select, "publish": publish,
     "final": final, "card": card, "report": report}[args.command](args)


if __name__ == "__main__":
    # As translate.py: a thread that is no daemon would keep a failed process alive and holding the TPU, so the
    # process ends here with the error's traceback and code
    import traceback

    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
