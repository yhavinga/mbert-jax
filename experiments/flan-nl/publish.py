#!/usr/bin/env python3
"""FLAN-nl on Hugging Face: the Dutch FLAN rows of arm F-nl, each traceable to its row in Open-Orca/FLAN.

Steps (commands.txt, step 11):
    link    on worker 0, CPU and network: every FLAN row of F's mixes (the English extracts en/w<h>.parquet) -> the file
            and row of Open-Orca/FLAN it was copied from -> links/w<h>.parquet and links/report.json
    build   after the assembly: nl/w<h>.parquet, links/, plan2/examples.parquet and nl/report.json -> a folder with
            data/train-*.parquet and README.md (the dataset card)
    upload  that folder to a public dataset repo

Open-Orca/FLAN has no id column: (revision, file, row) is the id. link reads per host the shards F's builder read
(build_decay_mix.select_flan: host_units of each folder, in shard order, as many as F's manifest says it read) and
matches on (task, inputs, targets), the strings F copied from those rows. Rows of identical content take the identical
source rows one by one in shard order, so each source row is given to at most one row of F.

The English text stays out of the dataset: build reads only the Dutch rows and the links.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
# flan_nl.py beside this file; F's builder in the repo's scripts/, or beside this file where it is copied to a VM
for _path in (HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

import build_decay_mix as bdm  # noqa: E402
import flan_nl  # noqa: E402

FLAN_REPO = bdm.FLAN_REPO
FLAN_REVISION = bdm.FLAN_REVISION
FLAN_FOLDERS = bdm.FLAN_FOLDERS
SOURCE_COLUMNS = ["inputs", "targets", "_task_name", "_template_idx", "_task_source", "_template_type"]
LINK_COLUMNS = ["host", "position", "flan_file", "flan_row", "template_idx", "task_source", "task_name", "template_type"]
DATA_COLUMNS = ["id", "inputs", "targets", "translated", "task_name", "task_source", "folder", "template_type",
                "template_idx", "flan_revision", "flan_file", "flan_row"]
LINK_METADATA_KEY = b"flan_nl_link"
CODE_URL = "https://github.com/yhavinga/mbert-jax/tree/flan-nl/experiments/flan-nl"


def folder_of(source):
    """'flan/<folder>/<task>' -> folder"""
    flan_nl.task_of(source)
    return source.split("/", 2)[1]


# ------------------------------------------------------------------------------------------------------------- link

class HubShards(bdm.HubShards):
    """F's listing, order and download of the shards (build_decay_mix.HubShards), read with the columns of a link."""

    def tables(self, names):
        """(name, table) in the order of names, a few downloaded ahead, each file deleted once read."""
        import concurrent.futures

        import pyarrow.parquet as pq

        names = list(names)
        executor = concurrent.futures.ThreadPoolExecutor(self.ahead)
        pending = deque(executor.submit(self._download, name) for name in names[:self.ahead])
        try:
            for k, name in enumerate(names):
                path = pending.popleft().result()
                if k + self.ahead < len(names):
                    pending.append(executor.submit(self._download, names[k + self.ahead]))
                table = pq.read_table(path, columns=SOURCE_COLUMNS)
                os.remove(path)
                yield name, table
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for future in pending:
                if future.done() and not future.cancelled() and future.exception() is None and os.path.exists(future.result()):
                    os.remove(future.result())


def match_shard(needed, table, residue=None, world=1, inputs=None):
    """[(id, row, template_idx, task_source, task_name, template_type)]: the rows of one shard that rows of needed are.

    needed: {(task, inputs, targets): deque of the ids still without a source row, first in line first}. A match takes
    the first id of its deque, so each row of the shard goes to at most one id, and identical rows of F take identical
    rows of the shard one by one. residue: the host read only rows residue, residue + world, ... of the shard (a folder
    of fewer shards than hosts, build_decay_mix.host_units). inputs: the inputs of needed as an arrow array, made once
    per folder; rows whose inputs are not among them are never turned into Python strings (a shard holds ~300,000)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if inputs is None:
        inputs = pa.array(sorted({x for _, x, _ in needed}), pa.string())
    candidate = pc.fill_null(pc.is_in(table.column("inputs"), value_set=inputs), False).to_numpy()
    if residue is not None:
        own = np.zeros(len(candidate), bool)
        own[residue::world] = True
        candidate = candidate & own
    rows = np.flatnonzero(candidate)
    if not len(rows):
        return []
    picked = table.take(pa.array(rows))
    out = []
    for row, x, y, task, template_idx, task_source, template_type in zip(
            rows.tolist(), *(picked.column(c).to_pylist() for c in SOURCE_COLUMNS)):
        waiting = needed.get((task, x, y))
        if waiting:
            out.append((waiting.popleft(), row, template_idx, task_source, task, template_type))
    return out


def link_host(host, english, manifest, source, log=print):
    """One host's FLAN rows of F linked to their source rows: (table of LINK_COLUMNS in position order, report)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.time()
    table = pq.read_table(english, columns=["host", "position", "source", "inputs", "targets"])
    build = pq.ParquetFile(english).schema_arrow.metadata[bdm.METADATA_KEY].decode()
    config, flan = manifest["config"], manifest["flan"]
    if (manifest["build"], config["rank"], config["flan"], list(config["folders"])) != \
            (build, host, f"{FLAN_REPO}@{FLAN_REVISION}", list(FLAN_FOLDERS)):
        raise SystemExit(f"{english} is host {host}'s extract of F's build {build}; the manifest is of build "
                         f"{manifest['build']}, rank {config['rank']}, {config['flan']}, folders {config['folders']}")
    hosts = set(table.column("host").to_pylist())
    if hosts != {host}:
        raise SystemExit(f"{english}: rows of hosts {sorted(hosts)}, expected only {host}")
    world = config["world"]
    needed = {folder: {} for folder in FLAN_FOLDERS}
    sharing = Counter()
    for position, source_name, x, y in zip(*(table.column(c).to_pylist() for c in ("position", "source", "inputs", "targets"))):
        folder = folder_of(source_name)
        if folder not in needed:
            raise SystemExit(f"{english}: position {position} is of {source_name}, not of F's folders {FLAN_FOLDERS}")
        key = (flan_nl.task_of(source_name), x, y)
        needed[folder].setdefault(key, deque()).append(position)
    links, folders = {}, {}
    for folder in FLAN_FOLDERS:
        wanted = needed[folder]
        sharing[folder] = sum(len(ids) for ids in wanted.values() if len(ids) > 1)
        left = sum(len(ids) for ids in wanted.values())
        report = flan["folders"][folder]
        names = source.shards(folder)
        if len(names) != report["shards"]:
            raise SystemExit(f"{FLAN_REPO}@{FLAN_REVISION} has {len(names)} shards in {folder}, F's manifest {report['shards']}")
        # The shards of this host that F read, in F's order: F stopped early once every task was at its cap
        units = bdm.host_units(len(names), host, world)[:report["shards_read"]]
        folders[folder] = {"rows": left, "rows_sharing_content": sharing[folder], "shards": len(names),
                           "shards_read_by_f": report["shards_read"], "shards_opened": 0,
                           "f_dropped": report["dropped"], "f_excluded_tasks": report["excluded_tasks"]}
        if not left:
            continue
        inputs = pa.array(sorted({x for _, x, _ in wanted}), pa.string())
        tables = source.tables([names[index] for index, _ in units])
        try:
            for (index, residue), (name, shard) in zip(units, tables):
                if name != names[index]:
                    raise RuntimeError(f"asked for {names[index]}, got {name}")
                types = set(shard.column("_template_type").unique().to_pylist())
                if types != {"zs_opt" if "_zsopt_" in folder else "zs_noopt"}:
                    raise SystemExit(f"{name} has template types {types}, not those of {folder}")
                matched = match_shard(wanted, shard, residue, world, inputs)
                for position, *rest in matched:
                    links[position] = (name, *rest)
                left -= len(matched)
                folders[folder]["shards_opened"] += 1
                log(f"link: host {host} {name} ({folders[folder]['shards_opened']} of {len(units)}): {len(matched):,} "
                    f"of {shard.num_rows:,} rows matched, {left:,} of {folders[folder]['rows']:,} left "
                    f"({time.time() - started:.0f} s)", flush=True)
                if not left:
                    break
        finally:
            tables.close()
        if left:
            missing = [(ids[0], task, x[:120], y[:60]) for (task, x, y), ids in wanted.items() if ids][:5]
            raise SystemExit(f"host {host}, {folder}: {left:,} rows of F have no source row in the "
                             f"{folders[folder]['shards_opened']} shards F read, e.g. (position, task, inputs, targets) "
                             f"{missing}")
    positions = sorted(links)
    if len(positions) != table.num_rows:
        raise RuntimeError(f"host {host}: {len(positions):,} links for {table.num_rows:,} rows")
    columns = list(zip(*(links[p] for p in positions)))
    out = pa.table({
        "host": pa.array([host] * len(positions), pa.int64()), "position": pa.array(positions, pa.int64()),
        "flan_file": pa.array(columns[0], pa.string()), "flan_row": pa.array(columns[1], pa.int64()),
        "template_idx": pa.array(columns[2], pa.int64()), "task_source": pa.array(columns[3], pa.string()),
        "task_name": pa.array(columns[4], pa.string()), "template_type": pa.array(columns[5], pa.string()),
    })
    report = {"f_build": build, "rows": table.num_rows, "linked": len(positions),
              "rows_sharing_content": sum(sharing.values()), "seconds": round(time.time() - started),
              "f_world": world, "f_cap": config["cap"], "f_cap_per_host": flan["per_host_cap"], "f_max_length": config["max_length"],
              "f_flan_share": config["flan_share"], "f_pool": flan["pool_examples"], "f_examples": flan["examples"],
              "folders": folders}
    return out, report


def link(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # The translation runs on this host's TPU: few threads, and the process runs under nice
    pa.set_cpu_count(args.arrow_threads)
    pa.set_io_thread_count(args.arrow_threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = HubShards(args.download_dir, ahead=args.ahead)
    for host in args.hosts:
        manifest = json.loads(Path(args.manifests, f"w{host}.json").read_text())
        table, report = link_host(host, Path(args.english, f"w{host}.parquet"), manifest, source)
        path = out / f"w{host}.parquet"
        partial = path.with_suffix(".partial")
        pq.write_table(table.replace_schema_metadata({LINK_METADATA_KEY: json.dumps(report).encode()}), partial,
                       compression="zstd")
        partial.rename(path)
        print(f"LINKED {json.dumps({'host': host, **{k: report[k] for k in ('rows', 'linked', 'rows_sharing_content', 'seconds')}})}",
              flush=True)
    # Of every host linked so far, also by an earlier run; the tasks F excluded once, not per host and folder
    hosts, excluded = {}, {}
    for path in sorted(out.glob("w*.parquet")):
        host = int(path.stem[1:])
        hosts[host] = json.loads(pq.ParquetFile(path).schema_arrow.metadata[LINK_METADATA_KEY])
        for folder in hosts[host]["folders"].values():
            for task, why in folder.pop("f_excluded_tasks").items():
                excluded.setdefault(why, set()).add(task)
    summary = {"flan": f"{FLAN_REPO}@{FLAN_REVISION}", "rows": sum(r["rows"] for r in hosts.values()),
               "linked": sum(r["linked"] for r in hosts.values()), "hosts": hosts,
               "f_excluded_tasks": {why: sorted(tasks) for why, tasks in sorted(excluded.items())}}
    (out / "report.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"done: {summary['linked']:,} of {summary['rows']:,} rows of hosts {sorted(hosts)} linked -> {out}", flush=True)


# ------------------------------------------------------------------------------------------------------------ build

# Every key of nl/report.json's hosts that counts rows left out, and what it means; a key not here stops the build, so
# the card never shows a number it cannot explain
DROPPED = {
    "dropped:task:code": "code, SQL, logical forms and other formal languages",
    "dropped:task:english_form": "the label depends on the English form: grammar, acceptability, spelling, punctuation, "
                                 "case, fluency, word order (CoLA, BLiMP, JFLEG, `fix_punct`, `true_case`, `word_segment`, ...)",
    "dropped:task:word_game": "word games, letters and strings (rhymes, character counts, lists of random letter strings, ...)",
    "dropped:task:token_level": "labels per token (POS tagging, NER) or the next or missing English word (LAMBADA, ...)",
    "dropped:task:translation": "translation, tokenization, or text in other languages",
    "dropped:task:english_words": "about English words: their senses (WiC, XL-WiC), the English language, overlapping words",
    "dropped:task:word_relations": "relations between bare words (odd man out, analogies, hypernyms, meronyms, antonyms, "
                                   "semantic relations): a word translated out of context can lose the sense the answer rests on",
    "dropped:segment_too_long": "a sentence of more than 512 tokens (TranslateGemma's tokenizer)",
    "dropped:segment_truncated": "a translation did not end within the decode limit",
    "dropped:segment_empty": "a translation is empty",
    "dropped:segment_length_ratio": "a translation has an implausible length: outside 0.5-2.2 times the source's characters "
                                    "(sources of 30 characters or more), or more than 3 times plus 30",
    "dropped:segment_commentary": "a translation carries a translator's commentary (\"Hier is de vertaling\", \"Vertaling:\", ...)",
    "dropped:segment_alternatives": "a translation offers alternatives (\" / \", or \";\" and \"/\" after a source of at most 4 words)",
    "dropped:segment_foreign_script": "a translation has letters of a script the source does not have (Devanagari, CJK, Cyrillic, ...)",
    "dropped:segment_not_dutch": "a translation is not Dutch (`langid` 1.1.6 over nl, en, de, fr, es, it, pt, af says "
                                 "neither nl nor af, from 5 words on, and the output has English function words in it)",
    "dropped:answer_not_in_instruction": "extractive answers: the English answer is a stretch of the English instruction, "
                                         "the Dutch answer is not a stretch of the Dutch instruction",
    "dropped:empty": "the Dutch instruction or answer is empty",
}
COUNTED = {"flan_rows", "translated", "kept_dutch_task"}
SEGMENT_CHECKS = {k.removeprefix("dropped:segment_"): v for k, v in DROPPED.items()
                  if k.startswith("dropped:segment_") and k != "dropped:segment_too_long"}
F_DROPPED = {
    "dumb_leak": "tasks whose name contains `sick` or `dbrd` (SICK, of which SICK-NL is a translation, and DBRD: "
                 "evaluation data of the experiment)",
    "eval_leak": "tasks whose name contains `belebele` or `flores` (evaluation data; Belebele's passages are FLORES-200)",
    "gpt_generated": "HellaSwag and MOCHA tasks, whose answer options were generated by GPT models",
    "language": "translation tasks of the FLAN 2021 folders (`wmt*`, `para_crawl*`) and NIv2 tasks with an instruction, "
                "input or output in a language other than English or Dutch",
    "unknown_character": "rows with `⁇` (U+2047, a character the T5 tokenizer could not represent)",
    "empty": "rows with an empty instruction or answer",
    "too_long": "rows of more than {max_length:,} tokens (instruction and answer, in the experiment's tokenizer)",
}


def read_links(paths):
    """{(host, position): (flan_file, flan_row, template_idx, task_source, task_name, template_type)}"""
    import pyarrow.parquet as pq

    links = {}
    for path in paths:
        table = pq.read_table(path, columns=LINK_COLUMNS)
        for h, p, *rest in zip(*(table.column(c).to_pylist() for c in LINK_COLUMNS)):
            if (h, p) in links:
                raise SystemExit(f"{path}: host {h}, position {p} is linked twice")
            links[(h, p)] = tuple(rest)
    return links


def size_category(n):
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M"),
                        (10_000_000, "1M<n<10M")):
        if n < limit:
            return name
    return "10M<n<100M"


def build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: an old shard there would be uploaded with the new ones")
    nl_report = json.loads(Path(args.nl, "report.json").read_text())
    link_report = json.loads(Path(args.links, "report.json").read_text())
    hosts = sorted(int(h) for h in nl_report["hosts"])
    if sorted(int(h) for h in link_report["hosts"]) != hosts:
        raise SystemExit(f"{args.links} has the links of hosts {sorted(link_report['hosts'])}, the assembly hosts {hosts}")
    links = read_links([Path(args.links, f"w{h}.parquet") for h in hosts])
    plan = pq.read_table(Path(args.plan, "examples.parquet"), columns=["host", "position", "status"])
    status = dict(zip(zip(plan.column("host").to_pylist(), plan.column("position").to_pylist()),
                      plan.column("status").to_pylist()))

    meta = {c: [] for c in DATA_COLUMNS if c not in ("inputs", "targets")}
    texts, missing, wrong = [], [], []
    for h in hosts:
        nl = pq.read_table(Path(args.nl, f"w{h}.parquet"), columns=["position", "source", "inputs", "targets"])
        for p, source in zip(nl.column("position").to_pylist(), nl.column("source").to_pylist()):
            link = links.get((h, p))
            if link is None:
                missing.append((h, p, source))
                continue
            flan_file, flan_row, template_idx, task_source, task_name, template_type = link
            folder = flan_file.split("/", 1)[0]
            if f"flan/{folder}/{task_name}" != source or status.get((h, p)) not in ("translate", "dutch_task"):
                wrong.append((h, p, source, flan_file, task_name, status.get((h, p))))
                continue
            for key, value in (("id", f"w{h}-{p}"), ("translated", status[(h, p)] != "dutch_task"),
                               ("task_name", task_name), ("task_source", task_source), ("folder", folder),
                               ("template_type", template_type), ("template_idx", template_idx),
                               ("flan_revision", FLAN_REVISION), ("flan_file", flan_file), ("flan_row", flan_row)):
                meta[key].append(value)
        texts.append(nl.select(["inputs", "targets"]))
    if missing:
        raise SystemExit(f"{len(missing):,} Dutch rows have no link to Open-Orca/FLAN, e.g. (host, position, source) {missing[:5]}")
    if wrong:
        raise SystemExit(f"{len(wrong):,} Dutch rows disagree with their link or the plan, e.g. (host, position, source, "
                         f"flan_file, task_name, status) {wrong[:5]}")
    text = pa.concat_tables(texts)
    types = {"id": pa.string(), "translated": pa.bool_(), "template_idx": pa.int64(), "flan_row": pa.int64()}
    data = pa.table({c: text.column(c) if c in ("inputs", "targets") else pa.array(meta[c], types.get(c, pa.string()))
                     for c in DATA_COLUMNS})
    if len(set(meta["id"])) != data.num_rows:
        raise SystemExit("ids are not unique")
    if len(set(zip(meta["flan_file"], meta["flan_row"]))) != data.num_rows:
        raise SystemExit("two Dutch rows have the same source row")
    data = data.sort_by([("flan_file", "ascending"), ("flan_row", "ascending")])

    # Arrow's in-memory size bounds the compressed file's, so every shard stays under the limit
    n = max(1, math.ceil(data.nbytes / args.shard_bytes))
    facts = {
        "rows": data.num_rows, "translated": sum(meta["translated"]), "files": n, "bytes": 0,
        "folders": Counter(meta["folder"]), "task_sources": Counter(meta["task_source"]),
        "folder_types": dict(zip(meta["folder"], meta["template_type"])),
        "tasks": len(set(zip(meta["folder"], meta["task_name"]))), "task_names": len(set(meta["task_name"])),
        "folder_tasks": Counter(f for f, _ in set(zip(meta["folder"], meta["task_name"]))),
        "dutch_tasks": sorted({t for t, kept in zip(meta["task_name"], meta["translated"]) if not kept}),
    }
    # The card checks the reports against the data: all of that before anything is written, so a failed build leaves
    # no folder to clear
    card(facts, nl_report, link_report, args.repo)
    (out / "data").mkdir(parents=True, exist_ok=True)
    per = math.ceil(data.num_rows / n)
    for k in range(n):
        path = out / "data" / f"train-{k:05d}-of-{n:05d}.parquet"
        pq.write_table(data.slice(k * per, per), path, compression="zstd", row_group_size=args.row_group_rows)
        if path.stat().st_size >= args.shard_bytes:
            raise SystemExit(f"{path} has {path.stat().st_size:,} bytes, over {args.shard_bytes:,}")
        facts["bytes"] += path.stat().st_size
    (out / "README.md").write_text(card(facts, nl_report, link_report, args.repo))
    print(f"BUILD {json.dumps({k: facts[k] for k in ('rows', 'translated', 'files', 'bytes', 'tasks')})} -> {out}", flush=True)


def _table(header, rows):
    """A markdown table; columns of counts ("1,234", "21 of 83") are aligned right."""
    rows = [[str(c) for c in row] for row in rows]
    numeric = [bool(rows) and all(re.fullmatch(r"[\d,]+( of [\d,]+)?", row[k]) for row in rows) for k in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def card(facts, nl_report, link_report, repo):
    """The dataset card: README.md with its YAML header, every number from the data and the two reports."""
    hosts = Counter()
    for counts in nl_report["hosts"].values():
        hosts.update(counts)
    unknown = sorted(k for k in hosts if k not in COUNTED and k not in DROPPED)
    if unknown:
        raise SystemExit(f"nl/report.json counts rows as {unknown}, which publish.py does not describe (DROPPED)")
    unusable = nl_report["segments_unusable"]
    if set(unusable) - set(SEGMENT_CHECKS):
        raise SystemExit(f"nl/report.json has segment checks {sorted(set(unusable) - set(SEGMENT_CHECKS))} not in SEGMENT_CHECKS")
    rows, translated = facts["rows"], facts["translated"]
    kept = rows - translated
    if (hosts["translated"], hosts["kept_dutch_task"]) != (translated, kept):
        raise SystemExit(f"nl/report.json counts {hosts['translated']:,} translated and {hosts['kept_dutch_task']:,} "
                         f"Dutch-task rows, the data {translated:,} and {kept:,}")
    f_rows = hosts["flan_rows"]
    if f_rows != link_report["rows"]:
        raise SystemExit(f"nl/report.json counts {f_rows:,} FLAN rows of F, links/report.json {link_report['rows']:,}")
    linked = list(link_report["hosts"].values())
    first = linked[0]
    pool = sum(r["f_pool"] for r in linked)
    # Distinct shards: in a folder of fewer shards than hosts every host reads every shard, each its own rows
    shards = {f: len({index for h, r in link_report["hosts"].items()
                      for index, _ in bdm.host_units(r["folders"][f]["shards"], int(h), r["f_world"])[:r["folders"][f]["shards_read_by_f"]]})
              for f in FLAN_FOLDERS}
    shards_total = {f: first["folders"][f]["shards"] for f in FLAN_FOLDERS}
    f_dropped = Counter()
    for r in linked:
        for f in FLAN_FOLDERS:
            f_dropped.update(r["folders"][f]["f_dropped"])
    excluded = link_report["f_excluded_tasks"]
    unknown = sorted(set(f_dropped) - set(F_DROPPED))
    if unknown:
        raise SystemExit(f"F's manifests drop rows as {unknown}, which publish.py does not describe (F_DROPPED)")
    repaired = nl_report["segments_repaired"]
    registers = {k: v for k, v in nl_report["template_registers"].items() if k != "rows with a variant outside the register"}
    with_templates = sum(registers.values())
    register_rows = Counter()
    for k, v in registers.items():
        register_rows[k.split()[0]] += v
    variants = nl_report["template_variants"]

    def task_rows(reason):
        return hosts.get(f"dropped:task:{reason}", 0)

    folder_rows = [(f"`{f}`", f"`{facts['folder_types'].get(f, '')}`", f"{facts['folders'][f]:,}", f"{facts['folder_tasks'][f]:,}",
                    f"{shards[f]} of {shards_total[f]}") for f in FLAN_FOLDERS]
    source_rows = [(f"`{s}`", f"{n:,}") for s, n in sorted(facts["task_sources"].items())]
    f_drop_rows = []
    for why in F_DROPPED:
        tasks = excluded.get(why, [])
        what = F_DROPPED[why].format(max_length=first["f_max_length"])
        if tasks:
            # Few tasks are named; the language filter excludes hundreds
            names = ": " + ", ".join(f"`{t}`" for t in tasks) if len(tasks) <= 12 else ""
            what += f" ({len(tasks):,} task{'s' if len(tasks) > 1 else ''}{names})"
        f_drop_rows.append((what, f"{f_dropped.get(why, 0):,}"))
    task_drop_rows = [(DROPPED[f"dropped:task:{reason}"], f"{len(names):,}", f"{task_rows(reason):,}")
                      for reason, names in flan_nl.DROPPED_TASKS.items()]
    after_rows = [(DROPPED[k], f"{hosts[k]:,}") for k in DROPPED if not k.startswith("dropped:task:") and hosts.get(k)]
    segment_rows = [(f"`{k}`", v, f"{unusable.get(k, 0):,}") for k, v in SEGMENT_CHECKS.items()]
    variant_rows = []
    for english, counts in variants.items():
        used = "; ".join(f"`{v}` ({n:,})" for v, n in counts.items()).replace("|", "\\|")
        variant_rows.append((f"`{english}`".replace("|", "\\|"), used))
    head_tail = "; ".join(f"`{k}` {n:,}" for k, n in nl_report["head_tail_pairs"].items())
    dropped_total = sum(v for k, v in hosts.items() if k.startswith("dropped:"))
    if f_rows - dropped_total != rows:
        raise SystemExit(f"nl/report.json: {f_rows:,} FLAN rows minus {dropped_total:,} dropped is not the {rows:,} rows of the data")

    yaml = "\n".join([
        "---", "license: cc-by-4.0", "language:", "- nl", "source_datasets:", f"- {FLAN_REPO}", "task_categories:",
        "- text-generation", "- text2text-generation", "size_categories:", f"- {size_category(rows)}", "pretty_name: FLAN-nl",
        "configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: data/train-*", "---"])
    return f"""{yaml}

# FLAN-nl

FLAN-nl is a Dutch machine translation of a sample of the [FLAN Collection](https://arxiv.org/abs/2301.13688) in its
zero-shot templates, as published in [{FLAN_REPO}](https://huggingface.co/datasets/{FLAN_REPO}). It is not a
translation of all of FLAN. It holds the FLAN rows that the FLAN arm of a Dutch ModernBERT experiment used:
{f_rows:,} English rows, of which {rows:,} are in this dataset ({translated:,} translated from English, {kept:,} of
Dutch tasks kept as they were). The other {f_rows - rows:,} were left out because their task does not survive
translation or their translation failed a check (see "Selection" and "Translation").

Every row names the file and the row of {FLAN_REPO} it comes from, at revision `{FLAN_REVISION}`, so that it can be
traced back to its English source.

## Contents

{_table(["", "rows"], [("translated from English (`translated = true`)", f"{translated:,}"),
                       ("Dutch tasks, not translated (`translated = false`)", f"{kept:,}"), ("total", f"{rows:,}")])}

{facts['tasks']:,} tasks (pairs of folder and task name; {facts['task_names']:,} task names), one split (`train`),
{facts['files']} parquet files ({facts['bytes'] / 1e6:,.0f} MB). `inputs` is the instruction (the prompt with its
template), `targets` the answer.

{_table(["folder of Open-Orca/FLAN", "template type", "rows", "tasks", "shards read"], folder_rows)}

{_table(["task source", "rows"], source_rows)}

## Selection

The rows were chosen in two stages: by the data builder of the experiment, from {FLAN_REPO}, and by the translation.

1. Source: the seven zero-shot folders of {FLAN_REPO} at revision `{FLAN_REVISION}`
   ({", ".join(f"`{f}`" for f in FLAN_FOLDERS)}). The few-shot folders were not used.
2. The builder split the source over {first["f_world"]} hosts: host h read shards h, h + {first["f_world"]}, ... of a
   folder (of a folder with fewer shards than hosts, rows h, h + {first["f_world"]}, ... of every shard), in the order of
   the shards, and left out:

{_table(["left out", "rows"], f_drop_rows)}

   The excluded tasks are counted over every row of the shards read, the last three filters only over the rows
   considered for a task that had not reached its cap yet. It took at most {first["f_cap_per_host"]:,} rows per (folder, task) per host, so at most {first["f_cap"]:,} per
   (folder, task) in all: {pool:,} rows.
3. Each host shuffled its rows with a fixed seed and used them until FLAN made up {100 * first["f_flan_share"]:.0f}% of
   the tokens of its training mix: {f_rows:,} rows. These are the rows that were translated.
4. Tasks that do not survive translation were left out (the lists `DROPPED_TASKS` in `flan_nl.py`):

{_table(["reason", "tasks", "rows"], task_drop_rows)}

5. Rows whose translation failed a check were left out:

{_table(["reason", "rows"], after_rows)}

## Translation

- Model: TranslateGemma 27B (`google/translategemma-27b-it` at revision `7d10f0b72f89a2d0f268cea30727d8b77c0d25c2`;
  [arXiv 2601.09012](https://arxiv.org/abs/2601.09012)), its language model converted to MaxText and run on TPU v4 in
  bf16, greedy decoding, with the model's chat template (source language `en`, target language `nl`).
- Segments: each instruction and answer was cut into segments, and each distinct segment was translated once
  ({nl_report["segments"]:,} segments). Multiple-choice options (labelled A/B/C, a/b/c, 1/2/3, i/ii/iii, `Option1:`,
  and bullet lists after a header such as `OPTIONS:`) are segments of their own; labels and separators were kept. An
  answer that is the text of an option got that option's translation, so it is still one of the options; an answer
  that is a label stayed the label. A segment of more than 512 tokens was cut at paragraphs, then lines, then
  sentences.
- Template strings: {len(variants)} fixed strings of FLAN's templates (labels such as `Q:`, `Answer:`, `Definition:`
  and opening sentences such as `Given the task definition and input, reply with output.`) were not machine
  translated but replaced by Dutch phrasings from a table with one or more variants per string, written by the
  dataset author with the help of Gemini. A row uses one form of address throughout: `je` or `u`, the one its machine
  translation uses, else one chosen by a hash of the row. Among the variants of that form, the one whose words the
  translation itself uses, else one chosen by hash. The labels `Head:` and `Tail:` follow the terms the translated
  task definition uses for them (for example `Beginzin:`/`Eindzin:` or `Kop:`/`Staart:`). {with_templates:,} rows have a
  template string; form of address: {", ".join(f"`{k}` {v:,}" for k, v in sorted(register_rows.items()))}.
- Repairs: the Devanagari danda (`।`) that the model sometimes writes for a period became a period
  ({repaired.get("danda", 0):,} segments); a final period that the source does not have was removed
  ({repaired.get("final_period", 0):,} segments).
- Checks: a translated segment was unusable, and every row with it left out, when:

{_table(["check", "meaning", "segments"], segment_rows)}

- Rows of the Dutch NIv2 tasks ({", ".join(f"`{t}`" for t in facts["dutch_tasks"])}) were kept as in the source: Dutch
  input and output with the English task definition of NIv2 (`translated = false`).

<details>
<summary>The template strings and the Dutch variants used (rows)</summary>

{_table(["English", "Dutch (rows)"], variant_rows)}

`Head:` and `Tail:`: {head_tail}

</details>

## Fields

{_table(["field", "type", "description"], [
        ("`id`", "string", "`w<host>-<position>`: the row's host and position in the experiment's training mix; unique"),
        ("`inputs`", "string", "the instruction, in Dutch"),
        ("`targets`", "string", "the answer, in Dutch"),
        ("`translated`", "bool", "false for the rows of Dutch tasks, kept as they were"),
        ("`task_name`", "string", "`_task_name` of the source row"),
        ("`task_source`", "string", "`_task_source` of the source row"),
        ("`folder`", "string", "the folder of the source file in Open-Orca/FLAN"),
        ("`template_type`", "string", "`_template_type` of the source row (`zs_opt`: with options, `zs_noopt`: without)"),
        ("`template_idx`", "int64", "`_template_idx` of the source row"),
        ("`flan_revision`", "string", "the revision of Open-Orca/FLAN"),
        ("`flan_file`", "string", "the parquet file of the source row in Open-Orca/FLAN"),
        ("`flan_row`", "int64", "the index of the source row in that file, from 0")])}

The rows are sorted by (`flan_file`, `flan_row`). Open-Orca/FLAN has no id column; (`flan_revision`, `flan_file`,
`flan_row`) identifies the source row.

## Tracing a row to its source

```python
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

row = load_dataset("{repo}", split="train")[0]
path = hf_hub_download("{FLAN_REPO}", row["flan_file"], repo_type="dataset", revision=row["flan_revision"])
source = pq.read_table(path).slice(row["flan_row"], 1).to_pylist()[0]
print(source["_task_name"], source["inputs"], source["targets"], sep="\\n")
```

## Limitations

- Machine translation, not reviewed by people. The checks above catch some failures; errors of meaning, wrong terms
  and literal renderings remain.
- The content is FLAN's: English-language sources with their cultural context (names, places, units, facts about the
  United States and other English-speaking countries) are translated, not adapted.
- Tasks about properties of English were left out as far as they are listed above, but some of the remaining tasks can
  read oddly after translation, for example where a question or an answer depends on English words, spelling or word
  order.
- The answers are translations, not Dutch originals.
- The cap per (folder, task) flattens FLAN's distribution of tasks: large tasks are sampled, small ones are taken whole.
- The rows of the Dutch tasks contain the English task definitions of NIv2.

## License

This dataset is released under CC BY 4.0, the license of {FLAN_REPO}. As there, the data is subject to the licenses of
the component datasets; the card of {FLAN_REPO} says:

> The data is subject to all the same licensing of the component datasets.

The translations were made with TranslateGemma, which is provided under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) (last modified April 1, 2026). Section 3.3:

> Google claims no rights in Outputs you generate using Gemma.

Section 1.1(f) defines Output as:

> the information content output of Gemma or a Model Derivative that results from operating or otherwise using Gemma or the Model Derivative, including via a Hosted Service.

If you train a model on this dataset, read the definition of "Model Derivatives" in Section 1.1(e):

> means all (i) modifications to Gemma, (ii) works based on Gemma, or (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Gemma, to that model in order to cause that model to perform similarly to Gemma, including distillation methods that use intermediate data representations or methods based on the generation of synthetic data Outputs by Gemma for training that model. For clarity, Outputs are not deemed Model Derivatives.

## Code

The code that selected, translated, assembled and published these rows:
[github.com/yhavinga/mbert-jax]({CODE_URL}), branch `flan-nl`, directory `experiments/flan-nl/` (`flan_nl.py`,
`translate.py`, `publish.py`); the selection of the FLAN rows is `scripts/build_decay_mix.py` of the same branch.

## Citation

The FLAN Collection, {FLAN_REPO} and TranslateGemma:

```bibtex
@misc{{longpre2023flan,
      title={{The Flan Collection: Designing Data and Methods for Effective Instruction Tuning}},
      author={{Shayne Longpre and Le Hou and Tu Vu and Albert Webson and Hyung Won Chung and Yi Tay and Denny Zhou and Quoc V. Le and Barret Zoph and Jason Wei and Adam Roberts}},
      year={{2023}},
      eprint={{2301.13688}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.AI}}
}}

@misc{{goodson2023huggyflan,
      title={{Fine FLAN: Seqio to Parquet So You Don't Have To}},
      author={{Bleys Goodson}},
      year={{2023}},
      publisher={{HuggingFace}},
      howpublished={{\\url{{https://huggingface.co/datasets/Open-Orca/FLAN}}}}
}}

@misc{{finkelstein2026translategemma,
      title={{TranslateGemma Technical Report}},
      author={{Mara Finkelstein and others}},
      year={{2026}},
      eprint={{2601.09012}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}}
}}
```

The component datasets of FLAN have their own citations; see the card of {FLAN_REPO}.
"""


# ----------------------------------------------------------------------------------------------------------- upload

def upload(args):
    from huggingface_hub import HfApi

    if not args.public:
        raise SystemExit("upload makes the dataset public: pass --public")
    folder = Path(args.folder)
    shards = sorted((folder / "data").glob("train-*.parquet"))
    if not (folder / "README.md").is_file() or not shards:
        raise SystemExit(f"{folder} needs README.md and data/train-*.parquet (publish.py build)")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(folder),
                      allow_patterns=["README.md", "data/train-*.parquet"], commit_message=args.message)
    print(f"https://huggingface.co/datasets/{args.repo}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="FLAN-nl on Hugging Face")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("link", help="every FLAN row of F -> its file and row in Open-Orca/FLAN")
    p.add_argument("--english", required=True, help="directory of the extracts w<h>.parquet (flan_nl.py extract)")
    p.add_argument("--manifests", required=True, help="directory of F's mix manifests w<h>.json ($R/../flan-decay/mix/)")
    p.add_argument("--out", required=True, help="directory of links/w<h>.parquet and report.json")
    p.add_argument("--hosts", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--download-dir", required=True, help="the shards while they are read; each is deleted after")
    p.add_argument("--ahead", type=int, default=3, help="shards downloaded ahead (default: 3)")
    p.add_argument("--arrow-threads", type=int, default=8)
    p = sub.add_parser("build", help="the folder for Hugging Face: data/train-*.parquet and README.md")
    p.add_argument("--nl", required=True, help="the assembly's directory: w<h>.parquet and report.json")
    p.add_argument("--links", required=True)
    p.add_argument("--plan", required=True, help="plan2/: examples.parquet, for which rows are of Dutch tasks")
    p.add_argument("--out", required=True, help="a new or empty directory")
    p.add_argument("--repo", default="yhavinga/flan-nl", help="the repo named in the card's example")
    p.add_argument("--shard-bytes", type=int, default=400_000_000)
    p.add_argument("--row-group-rows", type=int, default=20_000)
    p = sub.add_parser("upload", help="the folder to a public dataset repo")
    p.add_argument("--folder", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--public", action="store_true", help="required: the dataset is public")
    p.add_argument("--message", default="FLAN-nl: the Dutch FLAN rows of arm F-nl, with their source rows in Open-Orca/FLAN")
    args = parser.parse_args(argv)
    {"link": link, "build": build, "upload": upload}[args.step](args)


if __name__ == "__main__":
    main()
