#!/usr/bin/env python3
"""The main set of the en-nl translator up to its translation (hammond mbert/designs/en-nl-translator.md §3 step 0,
§7 point 6): the test sets as a block list, the pool's shards for the source check of Gemma 4 31B, and the check of
stage 0 v2 on the pilot.

    python mainset.py testsets --out <dir>
    python mainset.py judge-shards --select <dir> --out <dir> [--workers 4 --rows 2000]
    python mainset.py stage0-eval --annotated annotated.parquet --scores <source scores> --out stage0-eval.json
    python mainset.py eval-rows --plan <flan-nl plan2> --english <flan-nl en/w*.parquet> --out <dir>
    python mainset.py replay --plan <flan-nl plan2> --translations <translations/27b> --eval-rows <dir> --out <dir>
    python mainset.py report --select <pool dir> --stage0 stage0-eval.json --pilot-pairs pairs.parquet \
        --replay <dir> --eval-rows <dir> --testsets manifest.json [--scores <score dir>] --out mainset-prep.md

testsets: the texts of the test sets that no training passage may share a 13-gram with, one text a line, straight
from Hugging Face at fixed revisions (TESTSETS). They serve only as a list to block passages with, never as data.
judge-shards: passages.parquet and eval.parquet of a selection as shards of judge.py (id = pid, nl, en empty), every
worker an equal mix of lengths, with the passage's source, half, idiom class and document.
stage0-eval: per rule of stage 0 v2 (the fails of select_passages.py annotate), how many of the pilot's passages it
stops: among those Gemma 4 31B rejected on their source (score < 0.1) and among those it accepted (score >= 0.9).
eval-rows: the FLAN rows of arm F that §4 evaluates on, chosen first and fixed (design §7 point 6): 1,000 for the
classifier "original or translated" and 35 for Yeb's blind judgement, drawn with a fixed seed from the rows FLAN-nl
translated (status translate) whose English instruction has EVAL_BAND Gemma tokens (the band of the BT passages).
replay: the untagged part of arm N (§7 point 6): FLAN-nl at the level of its segments, an English segment and
TranslateGemma 27B's translation of it (the model's own input and output, before FLAN-nl's assembly). Filters in
order, each count reported: the segments of the eval rows, not ended, the Dutch outside the band of the BT passages,
decode_loop and english_left of publish.py, and a structure that differs (structure.py; TranslateGemma's empty line
for a single newline counted on its own).
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# select_passages.py and structure.py beside this file; publish.py (and what it imports) of FLAN-nl in the repo, or
# beside this file on a VM
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

# name: (repo, revision, file, how to read it, the fields that may hold text; those present are read, at least one
# must be). Dutch sides only: the passages are Dutch.
TESTSETS = {
    "dbrd_test": ("benjaminvdb/dbrd", "cb7fa0318d6b21d654d734a5a13fa52bd1033fc6",
                  "plain_text/test-00000-of-00001.parquet", "parquet", ["text"]),
    "sicknl_test": ("maximedb/sick_nl", "9648d68f54fe13d0372c3f89e365fec8e4c98856",
                    "data/test-00000-of-00001-2a492d0fcdd5479c.parquet", "parquet", ["sentence_A", "sentence_B"]),
    # facebook/flores and openlanguagedata/flores_plus are gated and Yeb's token is not admitted (9 Nov); the loading
    # scripts of FLORES-200 on the Hub (Muennighoff/flores200) read Meta's release itself, so that is read here, its
    # sha256 in the manifest
    "flores200_devtest_nld": ("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", None,
                              "./flores200_dataset/devtest/nld_Latn.devtest", "tar", None),
    "wmt24pp_en_nl": ("google/wmt24pp", "fd7405c06494bc66a57b25f55d217a72f96e60dc", "en-nl_NL.jsonl", "jsonl",
                      ["target", "original_target"]),
    "ntrex128_nld": ("davidstap/NTREX", "fd20d54141b6da952d5c68a2989472892489da0f",
                     "data/nld_Latn/newstest2019-ref.nld.txt", "lines", None),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def texts_of(path, kind, fields):
    """The texts of one test file, each on one line"""
    if kind == "lines":
        rows = [line.rstrip("\n") for line in open(path, encoding="utf-8")]
    elif kind == "jsonl":
        records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        present = [f for f in fields if any(f in r for r in records)]
        if not present:
            raise SystemExit(f"{path}: none of {fields} in its records ({sorted(records[0])})")
        rows = [r.get(f) for r in records for f in present]
    else:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        present = [f for f in fields if f in table.column_names]
        if not present:
            raise SystemExit(f"{path}: none of {fields} in its columns {table.column_names}")
        rows = [x for f in present for x in table.column(f).to_pylist()]
    return [" ".join(str(r).split()) for r in rows if r and str(r).strip()]


def testsets(args):
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, revision, filename, kind, fields) in TESTSETS.items():
        if kind == "tar":
            import tarfile
            import urllib.request

            archive = out / "download" / Path(repo).name
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                urllib.request.urlretrieve(repo, archive)
            with tarfile.open(archive) as tar:
                member = tar.extractfile(filename)
                if member is None:
                    raise SystemExit(f"{name}: no {filename} in {repo}")
                path = out / "download" / name / Path(filename).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(member.read())
            texts = texts_of(path, "lines", None)
            manifest[name] = {"url": repo, "file": filename, "texts": len(texts), "sha256_archive": sha256(archive),
                              "sha256": sha256(path)}
            (out / f"{name}.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
            print(f"{name}: {len(texts)} texts", flush=True)
            continue
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                               local_dir=str(out / "download" / name))
        texts = texts_of(path, kind, fields)
        if not texts:
            raise SystemExit(f"{name}: no texts in {repo}@{revision}/{filename}")
        (out / f"{name}.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        manifest[name] = {"repo": repo, "revision": revision, "file": filename, "fields": fields,
                          "texts": len(texts), "sha256": sha256(path)}
        print(f"{name}: {len(texts)} texts", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def judge_shards(args):
    """The pool as judge.py's input: <out>/w<k>/shard-NNNNN.parquet"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import select_passages as sp

    rows = []
    for name in ("passages.parquet", "eval.parquet"):
        path = Path(args.select) / name
        if path.exists():
            rows += pq.read_table(path).to_pylist()
    ids = [r["pid"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("a pid twice in the selection")
    out = Path(args.out)
    columns = ("id", "source", "split", "half", "idiom_class", "doc_id", "tokens", "nl", "en")
    counts = {}
    for k, part in enumerate(sp.balanced_workers([{**r, "id": r["pid"]} for r in rows], args.workers)):
        part.sort(key=lambda r: r["pid"])  # a shard holds a mix of lengths; judge.py batches by length
        d = out / f"w{k}"
        d.mkdir(parents=True, exist_ok=True)
        for s, start in enumerate(range(0, len(part), args.rows)):
            chunk = part[start:start + args.rows]
            table = {c: [r.get(c) for r in chunk] for c in columns if c not in ("nl", "en")}
            table["nl"] = [r["text"] for r in chunk]
            table["en"] = [""] * len(chunk)
            pq.write_table(pa.table(table), d / f"shard-{s:05d}.parquet", compression="zstd")
        counts[f"w{k}"] = len(part)
    print(f"JUDGE-SHARDS {len(rows)} passages: {counts}")


EVAL_BAND = (48, 384)  # select_passages.MIN_TOKENS, MAX_TOKENS
EVAL_SETS = (("classifier", 1000), ("yeb", 35))
EVAL_SEED = "en-nl-translator/flan-eval/1"


def segment_ids(pieces_json):
    """The segment ids of an instruction or answer of plan2's examples.parquet ([[kind, value], ...]; kind S)"""
    return [v for k, v in json.loads(pieces_json) if k == "S"]


def read_segments(plan):
    """{id: (English text, tokens)} of plan2's segments-w*.parquet"""
    import pyarrow.parquet as pq

    out = {}
    paths = sorted(Path(plan).glob("segments-w*.parquet"))
    if not paths:
        raise SystemExit(f"no segments-w*.parquet in {plan}")
    for path in paths:
        t = pq.read_table(path).to_pydict()
        out.update(zip(t["id"], zip(t["text"], t["tokens"])))
    return out


def eval_rows(args):
    import random

    import pyarrow as pa
    import pyarrow.parquet as pq

    examples = pq.read_table(Path(args.plan) / "examples.parquet",
                             columns=["host", "position", "source", "status", "instruction", "answer"]).to_pylist()
    segments = read_segments(args.plan)
    candidates = []
    for e in examples:
        if e["status"] != "translate":
            continue
        tokens = sum(segments[i][1] for i in segment_ids(e["instruction"]))
        if EVAL_BAND[0] <= tokens <= EVAL_BAND[1]:
            candidates.append((e["host"], e["position"], e["source"], tokens, e["instruction"], e["answer"]))
    candidates.sort()
    n = sum(k for _, k in EVAL_SETS)
    chosen = random.Random(EVAL_SEED).sample(candidates, n)
    english = {}
    for path in args.english:
        t = pq.read_table(path, columns=["host", "position", "inputs", "targets"]).to_pydict()
        wanted = {(h, p) for h, p, *_ in chosen}
        for h, p, i, o in zip(t["host"], t["position"], t["inputs"], t["targets"]):
            if (h, p) in wanted:
                english[(h, p)] = (i, o)
    missing = [(h, p) for h, p, *_ in chosen if (h, p) not in english]
    if missing:
        raise SystemExit(f"{len(missing)} chosen rows not in --english, e.g. {missing[:3]}")
    rows, start = [], 0
    for name, k in EVAL_SETS:
        for h, p, source, tokens, instruction, answer in chosen[start:start + k]:
            rows.append({"set": name, "host": h, "position": p, "source": source, "instruction_tokens": tokens,
                         "inputs": english[(h, p)][0], "targets": english[(h, p)][1],
                         "segments": segment_ids(instruction) + segment_ids(answer)})
        start += k
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in rows[0]}), out / "eval-rows.parquet")
    report = {"seed": EVAL_SEED, "band": EVAL_BAND, "examples": len(examples), "candidates": len(candidates),
              "sets": {name: k for name, k in EVAL_SETS},
              "segments": len({i for r in rows for i in r["segments"]}),
              "tasks": collections.Counter(r["source"] for r in rows).most_common(30)}
    (out / "eval-rows.json").write_text(json.dumps(report, indent=1))
    print(f"EVAL-ROWS {len(rows)} rows of {len(candidates)} candidates, {report['segments']} segments")


def _replay_checks(rows):
    import publish
    import structure

    out = []
    for en, nl in rows:
        out.append((publish.decode_loop(nl, en), publish.english_left(nl, en), structure.differences(en, nl)))
    return out


def replay(args):
    import concurrent.futures
    import multiprocessing

    import pyarrow as pa
    import pyarrow.parquet as pq

    segments = read_segments(args.plan)
    excluded = {i for s in pq.read_table(Path(args.eval_rows) / "eval-rows.parquet").column("segments").to_pylist()
                for i in s}
    translations = {}
    for path in sorted(Path(args.translations).glob("w*/*/shard-*.parquet")):
        t = pq.read_table(path, columns=["id", "text_nl", "ended", "tokens_out"]).to_pydict()
        for i, nl, ended, n in zip(t["id"], t["text_nl"], t["ended"], t["tokens_out"]):
            if i in translations:
                raise SystemExit(f"{path}: segment {i} translated twice")
            translations[i] = (nl, ended, n - int(ended))
    untranslated = [i for i in segments if i not in translations]
    if untranslated:
        raise SystemExit(f"{len(untranslated)} segments without a translation, e.g. {untranslated[:5]}")
    # a task per segment: the first example that has it (a segment can stand in several rows)
    examples = pq.read_table(Path(args.plan) / "examples.parquet", columns=["source", "instruction", "answer"])
    task = {}
    for source, ins, ans in zip(*(examples.column(c).to_pylist() for c in ("source", "instruction", "answer"))):
        for i in segment_ids(ins) + segment_ids(ans):
            task.setdefault(i, source)
    steps = collections.OrderedDict()
    live = sorted(segments)
    steps["segments"] = len(live)

    def keep(name, test):
        nonlocal live
        before = len(live)
        live = [i for i in live if test(i)]
        steps[name] = {"dropped": before - len(live), "left": len(live)}

    keep("eval_rows", lambda i: i not in excluded)
    keep("not_ended", lambda i: translations[i][1])
    keep("length_band", lambda i: EVAL_BAND[0] <= translations[i][2] <= EVAL_BAND[1])
    pairs = [(segments[i][0], translations[i][0]) for i in live]
    chunks = [pairs[k:k + 2000] for k in range(0, len(pairs), 2000)]
    if args.processes <= 1:
        checks = dict(zip(live, (c for part in map(_replay_checks, chunks) for c in part)))
    else:
        with concurrent.futures.ProcessPoolExecutor(args.processes,
                                                    mp_context=multiprocessing.get_context("spawn")) as pool:
            checks = dict(zip(live, (c for part in pool.map(_replay_checks, chunks) for c in part)))
    keep("decode_loop", lambda i: not checks[i][0])
    keep("english_left", lambda i: not checks[i][1])
    structure_counts = collections.Counter(k for i in live for k in checks[i][2])
    newline_doubled = sum(checks[i][2] == ["newline_doubled"] for i in live)
    keep("structure", lambda i: not checks[i][2])
    steps["structure"]["newline_doubled_only"] = newline_doubled
    steps["structure"]["by_category"] = dict(structure_counts)
    rows = [{"id": i, "task": task.get(i), "en": segments[i][0], "nl": translations[i][0],
             "tokens_en": segments[i][1], "tokens_nl": translations[i][2]} for i in live]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in ("id", "task", "en", "nl", "tokens_en", "tokens_nl")}),
                   out / "pairs.parquet", compression="zstd")
    # the dropped pairs of the structure check, a sample per category for reading
    import random

    dropped = [i for i in checks if checks[i][2] and not checks[i][0] and not checks[i][1]]
    sample = random.Random(0).sample(dropped, min(200, len(dropped)))
    pq.write_table(pa.table({"id": sample, "en": [segments[i][0] for i in sample],
                             "nl": [translations[i][0] for i in sample],
                             "differences": [checks[i][2] for i in sample]}), out / "structure-dropped.parquet")
    report = {"steps": steps, "band": EVAL_BAND, "excluded_segments": len(excluded),
              "tasks": collections.Counter(r["task"] for r in rows).most_common(40),
              "tokens_nl": sum(r["tokens_nl"] for r in rows), "tokens_en": sum(r["tokens_en"] for r in rows)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"REPLAY {json.dumps(steps)}")


RULES_NL = {
    "encoding": "U+FFFD, controle- of privétekens, of verkeerd gedecodeerde UTF-8 (Ã©, â€™)",
    "soft_hyphen": "een zacht afbreekstreepje (U+00AD) uit de PDF (\"burger\u00ad participatie\")",
    "hyphenation": "een woord met een afbreekstreepje dat de woordenlijst zonder streepje wel kent "
                   "(\"be-sloten\", \"Aanwij-zing\")",
    "sentence_start": "een alinea begint met een kleine letter, met een los pagina-, sectie- of lijstnummer "
                      "(\"27 Ik ben\", \"4.1.1 Op\"), of met een opsommingsteken in codeopmaak (`o`)",
    "cut_off": "de passage eindigt op een beletselteken (afgebroken spreekbeurt)",
    "lower_after_end": "na een punt gaat de tekst met een kleine letter verder (twee regels door elkaar)",
    "glued": "aan elkaar geplakte woorden (\"wasCrone\", \"GentMevrouw\", \"Voorzitter!Duivesteijn\")",
    "heading_glued": "een kop aan het begin van een regel zit vast aan de alinea (\"Burgernet Op 1 juli\")",
    "mid_capital": "verderop in een regel een functiewoord met hoofdletter zonder leesteken ervoor "
                   "(labels, paginanummers, kolommen: \"VVD 88 De\", \"extra Er zijn\")",
    "page_mark": "paginakop of -nummer in de tekst (\"Pagina 26 van 45\", \"15 mei 2025 2 Je\", \"- 13 De\", "
                 "\"besproken. 12 Het\")",
    "repeated": "een woordgroep die direct herhaald wordt (\"aan boord aan boord\")",
    "split_word": "een woord dat door een spatie in tweeën ligt (\"wan neer\" niet, \"zienswij zen\", \"fi ets\") "
                  "of losse letters (\"t oe a mbi\")",
    "open_before": "de regel vóór de passage eindigt op een woord dat nooit een zin eindigt (\"... van de\")",
    "continues_after": "de regel na de passage begint met een kleine letter (de laatste zin loopt door)",
}
STEP0_FILTERS = ("langid", "old_spelling", "oov", "digits", "punct", "capitals", "motion", "hyphenation",
                 "name_conflict", "speaker_change", "length")


def _md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" if k == 0 else "---:" for k in
                                                                       range(len(header))) + "|"]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def _n(x):
    return f"{x:,}".replace(",", ".") if isinstance(x, int) else x


def _quote(text, limit=700):
    text = text if len(text) <= limit else text[:limit] + " [...]"
    return "\n".join("> " + line for line in text.split("\n"))


def report(args):
    import random

    import pyarrow.parquet as pq

    import structure

    sel = json.loads((Path(args.select) / "report.json").read_text())
    stage0 = json.loads(Path(args.stage0).read_text())
    replay_report = json.loads((Path(args.replay) / "report.json").read_text())
    eval_report = json.loads((Path(args.eval_rows) / "eval-rows.json").read_text())
    testsets_manifest = json.loads(Path(args.testsets).read_text())
    sources = ["handelingen", "raad", "dpc", "rechtspraak"]
    out = ["# Hoofdset en→nl-vertaler: voorbereiding tot vlak vóór het vertalen", "",
           f"Gegenereerd door `mainset.py report`. Bucket: `gs://t5-dutch-english-usc2/mbert/translator/mainset/`. "
           f"Geen vertaling, geen drempels, geen fine-tune.", ""]

    # stage 0 v2 on the pilot
    sides = stage0["sides"]
    out += ["## 1. Stap 0 v2 op de 2.000 pilotpassages", "",
            f"Gemma 4 31B (bronprompt van `source-quality.md`) keurde {sides['rejected']['all']} passages af (score "
            f"< 0,1) en {sides['accepted']['all']} goed (≥ 0,9); {sides['between']['all']} zaten ertussen. Per regel: "
            f"hoeveel van die afgekeurde hij vangt en hoeveel goedgekeurde hij ten onrechte tegenhoudt (een passage kan "
            f"op meer regels vallen). De regels zijn opnieuw toegepast op de pilotdocumenten (`select_passages.py "
            f"annotate`, dezelfde row groups en seed); {len(stage0['text_changed'])} van de 2.000 teksten wijken af "
            f"door een wijziging van na de pilotselectie (afkortingen in de zinsplitsing, commit 9853d3c).", ""]
    rows = []
    for rule in [r for r in stage0["rules"] if r != "any"] + ["any"]:
        e = stage0["rules"][rule]
        rows.append([f"`{rule}`" if rule != "any" else "**een of meer**", e["rejected"]["all"], e["accepted"]["all"],
                     e["between"]["all"], e["only_rule_rejected"] if rule != "any" else "",
                     RULES_NL.get(rule, "filter van de pilot" if rule in STEP0_FILTERS else "")])
    out += [_md_table(["regel", f"gevangen van {sides['rejected']['all']} afgekeurd",
                       f"tegengehouden van {sides['accepted']['all']} goedgekeurd", "ertussen",
                       "alleen deze regel (afgekeurd)", "wat"], rows), ""]
    any_rule = stage0["rules"]["any"]
    out += ["Per bron (een of meer regels):", "",
            _md_table(["bron", "afgekeurd door Gemma", "daarvan gevangen", "goedgekeurd", "daarvan tegengehouden"],
                      [[s, sides["rejected"][s], any_rule["rejected"][s], sides["accepted"][s],
                        any_rule["accepted"][s]] for s in sources]), ""]

    # the pool
    out += ["## 2. Kandidatenpool", "",
            "Doel: na het Gemma-bronfilter 200.000 passages in de verdeling 50/30/10/10, met ~30% marge, gerekend met "
            "de afkeurpercentages van de pilot (score < 0,1). Plus 4.000 evaluatiepassages uit andere documenten "
            "(splitsing per document), met dezelfde marge.", ""]
    rates = {s: sides["rejected"][s] / (sides["rejected"][s] + sides["accepted"][s] + sides["between"][s])
             for s in sources}
    args_sel = sel["args"]
    counts = dict(part.split("=") for part in args_sel["counts"].split(",")) if args_sel.get("counts") else {}
    eval_counts = dict(part.split("=") for part in args_sel["eval_counts"].split(",")) \
        if args_sel.get("eval_counts") else {}
    out += [_md_table(["bron", "aandeel", "nodig", "afkeur pilot", "pool = nodig × 1,3 / (1 − afkeur)", "eval nodig",
                       "eval-pool"],
                      [[s, f"{int(100 * sh)}%", _n(int(200000 * sh)), f"{100 * rates[s]:.1f}%", _n(int(counts.get(s, 0))),
                        _n(int(4000 * sh)), _n(int(eval_counts.get(s, 0)))]
                       for s, sh in zip(sources, (0.5, 0.3, 0.1, 0.1))]), ""]
    out += [f"Selectie: `select_passages.py select` met `--row-groups {args_sel['row_groups']}`, `--max-per-doc "
            f"{args_sel['max_per_doc']}` (anders 3), `--eval-permille {args_sel['eval_permille']}`, seed "
            f"{args_sel['seed']}, dezelfde bronnen en revisies als de pilot; {sel['seconds']} s.", ""]
    rows = []
    for s in sources:
        r = sel["sources"][s]
        d, c = r["documents"], r["counts"]
        rows.append([s, r["row_groups"], _n(d.get("docs_read", 0)), _n(d.get("docs_kept", 0)), _n(c.get("passages", 0)),
                     _n(r["passed"]), _n(r["passed_train"]), _n(r["passed_eval"]), _n(r["chosen"]),
                     _n(r["chosen_eval"]), r["chosen_halves"].get("idiom", 0), r["choice"]["idiom_class_size"]])
    out += [_md_table(["bron", "row groups", "documenten", "na documentfilters", "passages", "door alle filters",
                       "train-kant", "eval-kant", "pool (train)", "pool (eval)", "waarvan idioomhelft",
                       "idioomklasse"], rows), "",
            f"Overgeslagen bij het kiezen (alle bronnen): {sel['skipped_while_choosing']}. Geblokkeerde 13-grammen "
            f"uit de testsets: {_n(sel.get('blocked_ngrams', 0))}. De idioomhelft is kleiner dan de helft waar de "
            f"bron te weinig passages met een idioomtreffer heeft; de rest komt dan uit de willekeurige helft.", ""]
    rules = [r for r in STEP0_FILTERS if r != "length"] + list(RULES_NL) + ["length"]
    rules = list(dict.fromkeys(rules))
    out += ["Afgevallen passages per filter (een passage kan op meer filters vallen):", "",
            _md_table(["filter"] + sources, [[f"`{f}`"] + [_n(sel["sources"][s]["fails"].get(f, 0)) for s in sources]
                                            for f in rules if any(sel["sources"][s]["fails"].get(f) for s in sources)]),
            ""]
    out += ["Testsets als uitsluitlijst (alleen 13-gramblokkade, nooit als data):", "",
            _md_table(["set", "bron", "teksten"],
                      [[k, f"`{v.get('repo', v.get('url'))}`" + (f" @ `{v['revision'][:7]}`" if v.get("revision") else ""),
                        v["texts"]] for k, v in testsets_manifest.items()]), "",
            "FLORES-200 komt uit Meta's release (`flores200_dataset.tar.gz`): `facebook/flores` en "
            "`openlanguagedata/flores_plus` zijn afgeschermd en Yebs token heeft er geen toegang toe.", ""]

    # source scores of the pool
    out += ["## 3. Bronscore van Gemma 4 31B in de pool", ""]
    pool = {}
    for name in ("passages.parquet", "eval.parquet"):
        path = Path(args.select) / name
        if path.exists():
            for r in pq.read_table(path, columns=["pid", "source", "split", "half", "text"]).to_pylist():
                pool[r["pid"]] = r
    scored = {}
    if args.scores:
        for path in sorted(Path(args.scores).glob("w*/shard-*.parquet")):
            for r in pq.read_table(path, columns=["id", "score", "logp_yes", "logp_no", "skipped"]).to_pylist():
                scored[r["id"]] = r
    if not scored:
        out += ["Nog niet gescoord: deze sectie vult `mainset.py report --scores` na de run.", ""]
    else:
        rows = []
        for s in sources:
            sc = [scored[p]["score"] for p in pool if pool[p]["source"] == s and p in scored
                  and scored[p]["skipped"] is None]
            lo = sorted(scored[p]["logp_yes"] - scored[p]["logp_no"] for p in pool if pool[p]["source"] == s
                        and p in scored and scored[p]["skipped"] is None)
            n = len(sc)
            if not n:
                continue
            bins = [sum(1 for x in sc if k / 10 <= x < (k + 1) / 10 or (k == 9 and x == 1.0)) for k in range(10)]
            rows.append([s, _n(n), f"{sum(sc) / n:.3f}", f"{100 * sum(x < 0.1 for x in sc) / n:.1f}%",
                         f"{100 * sum(x < 0.5 for x in sc) / n:.1f}%", f"{100 * sum(x >= 0.9 for x in sc) / n:.1f}%",
                         f"{lo[int(0.05 * (n - 1))]:.1f} / {lo[n // 2]:.1f} / {lo[int(0.95 * (n - 1))]:.1f}"]
                        + [_n(b) for b in bins])
        missing = len([p for p in pool if p not in scored])
        out += [f"{_n(len(scored))} van {_n(len(pool))} poolpassages gescoord ({_n(missing)} zonder score). Geen "
                f"drempel gekozen. Log-odds = log P(yes) − log P(no).", "",
                _md_table(["bron", "passages", "gemiddeld", "< 0,1", "< 0,5", "≥ 0,9", "log-odds p5 / p50 / p95"]
                          + [f"{k / 10:.1f}–{(k + 1) / 10:.1f}" for k in range(10)], rows), ""]
        out += ["### Voorbeelden per bron (5 met score < 0,1 en 5 met score ≥ 0,9, willekeurig, seed 0; tekst na 700 "
                "tekens afgekapt)", ""]
        for s in sources:
            r = random.Random(f"0/{s}")
            ids = sorted(p for p in pool if pool[p]["source"] == s and p in scored)
            low = [p for p in ids if scored[p]["score"] < 0.1]
            high = [p for p in ids if scored[p]["score"] >= 0.9]
            for label, group in (("afgekeurd", low), ("doorgelaten", high)):
                out += [f"#### {s}, {label}", ""]
                for p in r.sample(group, min(5, len(group))):
                    x = scored[p]
                    out += [f"**{x['score']:.4f}** (log-odds {x['logp_yes'] - x['logp_no']:.1f}) · `{p}` · "
                            f"{pool[p]['split']}, {pool[p]['half']}", "", _quote(pool[p]["text"]), ""]

    # FLAN-nl replay
    st = replay_report["steps"]
    out += ["## 4. FLAN-nl-replay (ongetagde deel van arm N)", "",
            f"Eerst vastgelegd: {eval_report['sets']['classifier']} FLAN-rijen voor de classifier en "
            f"{eval_report['sets']['yeb']} voor Yebs blinde beoordeling, uit de {_n(eval_report['candidates'])} rijen "
            f"van F die FLAN-nl vertaalde (`status translate`) met een Engelse instructie van {eval_report['band'][0]}–"
            f"{eval_report['band'][1]} Gemma-tokens, seed `{eval_report['seed']}`; samen {_n(eval_report['segments'])} "
            f"segmenten, die uit de replay blijven. Daarna de segmentparen (Engels segment → vertaling van "
            f"TranslateGemma 27B), met de filters in deze volgorde:", ""]
    rows = [["segmenten van plan2", "", _n(st["segments"])]]
    for name, label in (("eval_rows", "segment van een evaluatierij"), ("not_ended", "`ended` onwaar"),
                        ("length_band", f"Nederlands buiten {replay_report['band'][0]}–{replay_report['band'][1]} "
                                        f"tokens (de band van de BT-passages)"),
                        ("decode_loop", "`decode_loop` (publish.py)"), ("english_left", "`english_left` (publish.py)"),
                        ("structure", "structuur verschilt (structure.py)")):
        rows.append([label, _n(st[name]["dropped"]), _n(st[name]["left"])])
    out += [_md_table(["filter", "valt af", "blijft over"], rows), "",
            f"Van de {_n(st['structure']['dropped'])} paren met een ander structuur is bij "
            f"{_n(st['structure']['newline_doubled_only'])} het enige verschil de bekende afwijking van TranslateGemma "
            f"(een lege regel waar het Engels één regelovergang heeft). Per categorie (een paar kan in meer vallen): "
            f"{json.dumps(st['structure']['by_category'])}. Over: {_n(replay_report['tokens_en'])} Engelse en "
            f"{_n(replay_report['tokens_nl'])} Nederlandse tokens.", ""]

    # structure on the pilot pairs
    pilot = pq.read_table(args.pilot_pairs, columns=["source", "nl", "en", "back"]).to_pylist()
    out += ["## 5. Structuurcontrole op de 2.000 pilotparen", "",
            "Categorieën: koppen, opsommings- en nummeritems, vet, cursief, codeblokken, inline code, tabelrijen en "
            "-scheidingslijnen, alinea- en regelovergangen (`structure.py`). `newline_doubled` = alleen regelovergangen "
            "van de bron die in de vertaling witregels zijn.", ""]
    rows = []
    for label, a, b in (("nl → en (origineel → synthetisch Engels)", "nl", "en"),
                        ("en → nl (synthetisch Engels → terugvertaling)", "en", "back")):
        per_source, per_cat, total, only_doubled = collections.Counter(), collections.Counter(), 0, 0
        for r in pilot:
            d = structure.differences(r[a], r[b] or "")
            if d:
                total += 1
                per_source[r["source"]] += 1
                per_cat.update(d)
                only_doubled += d == [structure.NEWLINE_DOUBLED]
        rows.append([label, total, only_doubled, total - only_doubled,
                     ", ".join(f"{s} {per_source[s]}" for s in sources),
                     ", ".join(f"{k} {v}" for k, v in per_cat.most_common())])
    out += [_md_table(["richting", "paren met verschil", "alleen newline_doubled", "overige", "per bron",
                       "per categorie"], rows), ""]
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"REPORT {args.out}: {len(out)} lines, {len(scored)} scores")


SIDES = (("rejected", lambda s: s < 0.1), ("accepted", lambda s: s >= 0.9), ("between", lambda s: 0.1 <= s < 0.9))


def stage0_eval(args):
    """Per rule and source: passages stopped among Gemma's rejected and accepted ones, and in all"""
    import pyarrow.parquet as pq

    scores = {r["id"]: r["score"] for path in args.scores for r in pq.read_table(path).to_pylist()}
    rows = pq.read_table(args.annotated).to_pylist()
    texts = {}
    if args.passages:
        texts = {r["pid"]: r["text"] for r in pq.read_table(args.passages).to_pylist()}
    missing = [r["pid"] for r in rows if r["pid"] not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} passages without a source score, e.g. {missing[:3]}")
    changed = [r["pid"] for r in rows if texts and texts.get(r["pid"]) != r["text"]]
    rules = sorted({f for r in rows for f in r["fails"]})
    result = {"passages": len(rows), "text_changed": changed, "rules": {}, "sides": {}}
    groups = collections.defaultdict(list)
    for r in rows:
        side = next(name for name, test in SIDES if test(scores[r["pid"]]))
        groups[(side, r["source"])].append(r)
        groups[(side, "all")].append(r)
    sources = ["all"] + sorted({r["source"] for r in rows})
    for side, _ in SIDES:
        result["sides"][side] = {s: len(groups[(side, s)]) for s in sources}
    for rule in rules + ["any"]:
        entry = {}
        for side, _ in SIDES:
            entry[side] = {s: sum((bool(r["fails"]) if rule == "any" else rule in r["fails"]) for r in groups[(side, s)])
                           for s in sources}
        entry["only_rule_rejected"] = sum(r["fails"] == [rule] for r in groups[("rejected", "all")])
        result["rules"][rule] = entry
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"STAGE0-EVAL {len(rows)} passages, {len(changed)} with another text than the pilot's")
    for rule in rules + ["any"]:
        e = result["rules"][rule]
        print(f"  {rule:16} rejected {e['rejected']['all']:4} / {result['sides']['rejected']['all']}   accepted "
              f"{e['accepted']['all']:4} / {result['sides']['accepted']['all']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("testsets", help="The test sets' Dutch texts as a block list")
    p.add_argument("--out", required=True)
    p = sub.add_parser("judge-shards", help="A selection as judge.py's input shards per worker")
    p.add_argument("--select", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rows", type=int, default=2000)
    p = sub.add_parser("eval-rows", help="The FLAN rows of F that §4 evaluates on, fixed with a seed")
    p.add_argument("--plan", required=True)
    p.add_argument("--english", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("replay", help="FLAN-nl's segment pairs for the untagged part of arm N")
    p.add_argument("--plan", required=True)
    p.add_argument("--translations", required=True)
    p.add_argument("--eval-rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--processes", type=int, default=64)
    p = sub.add_parser("report", help="mainset-prep.md")
    p.add_argument("--select", required=True)
    p.add_argument("--stage0", required=True)
    p.add_argument("--pilot-pairs", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--eval-rows", required=True)
    p.add_argument("--testsets", required=True)
    p.add_argument("--scores", help="The local copy of the score shards (w*/shard-*.parquet)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("stage0-eval", help="Stage 0 v2's rules against Gemma's source scores of the pilot")
    p.add_argument("--annotated", required=True)
    p.add_argument("--scores", required=True, nargs="+")
    p.add_argument("--passages", help="The pilot's passages.parquet, to check that annotate rebuilt the same texts")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"testsets": testsets, "judge-shards": judge_shards, "stage0-eval": stage0_eval, "eval-rows": eval_rows,
     "replay": replay, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
