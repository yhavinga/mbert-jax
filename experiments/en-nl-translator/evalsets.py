#!/usr/bin/env python3
"""The evaluation sets of the en-nl translator (hammond mbert/designs/en-nl-translator.md §4) and the measurement of a
translator on them. These sets must never reach training data: everything of them lives under translator/eval/.

    python evalsets.py fetch --out <dir> --tokenizer <TranslateGemma tokenizer dir> [--workers 4]
    python evalsets.py collect --data <dir> --worker <k> --shards <local copy of worker k's shards> --out pairs-w<k>.parquet
    python evalsets.py summary --data <dir> --pairs <pairs-w*.parquet> --metricx <...> --comet <...> --out <dir>

fetch downloads the four sets from their own sources at fixed revisions and checks every file's sha256:
- wmt24pp: google/wmt24pp en-nl_NL.jsonl (998 paragraphs; reference: the post-edited `target`, as the dataset card
  recommends; `is_bad_source` and `original_target` are kept);
- flores: FLORES-200 devtest eng_Latn -> nld_Latn (1,012 sentences) from Meta's release, the tarball (facebook/flores
  on the Hub is gated for Yeb's token);
- ntrex: NTREX-128 newstest2019 eng -> nld (1,997 sentences) from GitHub MicrosoftTranslator/NTREX at a commit;
- europarl: Europarl-direct (Zenodo 5596238, MoTra 2021) multi8_{train,dev,test}.tsv, the 640 paragraphs with
  src=nl and dest=en: the English (`text`, translated directly from the Dutch) is the input, the Dutch original
  (`original`) the reference.
It writes evalsets.parquet (one row per segment: id, set, source, reference, tokens of the source under TranslateGemma's
tokenizer, over_512, length_class, meta as JSON), segments-w<k>.parquet (translate.py's input: id, text, tokens),
classes-w<k>.txt (the length classes a worker has) and manifest.json (sources, revisions, sha256, licences, counts).
A source above 768 tokens stops it (translate.py's longest class); one above 512 is marked over_512 (the class xxlong
went wrong in the main run's minipilot) and counted.

Workers get whole length classes (translate.py loads the model once per class, ~2-3.5 min, which is most of the time
for these few segments): the classes by estimated time, largest first, each to the worker with the least so far.

collect joins one worker's translations (translate.py's shards) with the sets: pairs-w<k>.parquet with source,
hypothesis, reference, ended, tokens, and FLAN-nl's decode_loop and english_left of the hypothesis against its source.
summary joins the pairs with MetricX-24 (reference and QE; metricx_qe.py) and COMET-22 (comet_score.py) and writes
translations.parquet, scores.parquet, summary.json and summary.md: per set the mean and median of each metric and the
number of segments that did not end, that loop, and that are over 512 tokens.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# translate.py of FLAN-nl and publish.py (and what it imports) in the repo, or beside this file on a VM
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

from translate import CLASS_ORDER, LENGTH_CLASSES  # noqa: E402

WMT24PP = ("google/wmt24pp", "fd7405c06494bc66a57b25f55d217a72f96e60dc")
NTREX_COMMIT = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
ZENODO = "https://zenodo.org/api/records/5596238/files/{}/content"
NTREX_URL = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/" + NTREX_COMMIT + "/{}"
SHA256 = {
    "en-nl_NL.jsonl": "98c8ff19e8c97fe4ccbc2ee0e46898e41cb7a17c81b4c9205c7f20286f82b4a8",
    "flores200_dataset.tar.gz": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
    "eng_Latn.devtest": "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88",
    "nld_Latn.devtest": "2b22cf9a1c2c6c80b8bf733e6da93c7017297d7557b67f7799613b150bb5c07f",
    "metadata_devtest.tsv": "8edac47f861fcf6b54dad2068c10ba8f741da8b8676861c6bc27fcd9f5483b36",
    "newstest2019-src.eng.txt": "389e8f5796c66db4f646dfad33e1ec622d74767af5ef112b42a1f2cd814df3cc",
    "newstest2019-ref.nld.txt": "dce60b592d410932cb369301d0506f1b3d0ae4f7bedf0369a3cdf870f462b4a5",
    "multi8_train.tsv": "c57462f2e0cb949c8adde5e1ca48870229b99e991a7d6029f3a79f59ed31c86a",
    "multi8_dev.tsv": "7f0ca82c01281f2274c8896984806bc62e37a5eeba05bd93de82da415cbfbfb4",
    "multi8_test.tsv": "8f6e8f6cca1c2fde0b45585a269a6abff736b6ac984de3d9ae32c04d0d55d383",
}
# Descriptions in Dutch: they go into summary.md, which is Yeb's report
SETS = {
    "wmt24pp": {
        "name": "WMT24++ en→nl_NL", "count": 998,
        "source": f"https://huggingface.co/datasets/{WMT24PP[0]} @ {WMT24PP[1]}, en-nl_NL.jsonl",
        "license": "Apache-2.0 (datasetkaart, `license: apache-2.0`)",
        "reference": "`target` (de post-edit; de datasetkaart raadt die als referentie aan); `original_target` in meta",
    },
    "flores": {
        "name": "FLORES-200 devtest eng_Latn→nld_Latn", "count": 1012,
        "source": FLORES_URL + " (devtest/eng_Latn.devtest, devtest/nld_Latn.devtest, metadata_devtest.tsv)",
        "license": "CC BY-SA 4.0 (github.com/facebookresearch/flores, README.md @ a6c830c: 'FLORES-200: CC-BY-SA 4.0')",
        "reference": "nld_Latn.devtest",
    },
    "ntrex": {
        "name": "NTREX-128 newstest2019 eng→nld", "count": 1997,
        "source": f"https://github.com/MicrosoftTranslator/NTREX @ {NTREX_COMMIT}, NTREX-128/newstest2019-src.eng.txt "
                  "en newstest2019-ref.nld.txt",
        "license": "CC BY-SA 4.0 (README.md en LICENSE.md van de repo)",
        "reference": "newstest2019-ref.nld.txt",
    },
    "europarl": {
        "name": "Europarl-direct nl→en, omgekeerd (het Nederlandse origineel als referentie)", "count": 640,
        "source": "https://zenodo.org/records/5596238 (versie 2.0.0, doi 10.5281/zenodo.5596238), "
                  "multi8_{train,dev,test}.tsv, rijen src=nl en dest=en",
        "license": "CC BY 4.0 (Zenodo-record); de teksten zijn handelingen van het Europees Parlement",
        "reference": "`original` (de toespraak zoals in het Nederlands gehouden); de invoer is `text`, de directe "
                     "Engelse vertaling daarvan",
    },
}
LOAD_SECONDS = 180  # translate.py's loading and compiling per length class (2-3.5 min in the pilot and main run)
OUT_PER_SECOND = 1200  # decode tokens per second of one host (main run, class small)
DUTCH_PER_ENGLISH = 1.3  # output tokens per source token, en->nl
MARK_OVER = 512


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, attempts=3):
    """url to path (through path.part); a network error is tried again, attempts times in all"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(f"{path}.part", "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            Path(f"{path}.part").rename(path)
            return path
        except OSError as e:
            if attempt == attempts:
                raise SystemExit(f"download of {url} failed {attempts} times, last: {e}")
            print(f"download of {url} failed ({e}); attempt {attempt + 1} of {attempts} in {30 * attempt} s",
                  flush=True)
            time.sleep(30 * attempt)


def checked(path):
    """path, after its sha256 matched the pinned one of its file name"""
    digest = sha256(path)
    if digest != SHA256[path.name]:
        raise SystemExit(f"{path}: sha256 {digest}, pinned {SHA256[path.name]}")
    return path


def lines(path, count):
    """The lines of a text file without their terminator (the NTREX files end lines in CRLF); split on newlines only,
    not on what str.splitlines also takes for a line end (U+2028, U+0085), so the sides stay aligned"""
    text = path.read_text(encoding="utf-8")
    out = text.split("\n")
    if out and out[-1] == "":
        out.pop()
    out = [x[:-1] if x.endswith("\r") else x for x in out]
    if len(out) != count:
        raise SystemExit(f"{path}: {len(out)} lines, expected {count}")
    return out


def read_wmt24pp(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return [{"source": r["source"], "reference": r["target"],
             "meta": {k: r[k] for k in ("domain", "document_id", "segment_id", "is_bad_source", "original_target")}}
            for r in records]


def read_flores(eng, nld, metadata):
    n = SETS["flores"]["count"]
    meta = lines(metadata, n + 1)
    header = meta[0].split("\t")
    rows = [dict(zip(header, m.split("\t"))) for m in meta[1:]]
    return [{"source": s, "reference": r, "meta": m} for s, r, m in zip(lines(eng, n), lines(nld, n), rows)]


def read_ntrex(src, ref):
    n = SETS["ntrex"]["count"]
    return [{"source": s, "reference": r, "meta": {"line": k}} for k, (s, r) in enumerate(zip(lines(src, n),
                                                                                              lines(ref, n)))]


def read_europarl(paths):
    """Rows src=nl, dest=en of the multi8 files (columns iid src native_speaker original dest text direct label)"""
    csv.field_size_limit(1 << 30)
    out = []
    for split, path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)
            if header != ["iid", "src", "native_speaker", "original", "dest", "text", "direct", "label"]:
                raise SystemExit(f"{path}: unexpected header {header}")
            for k, row in enumerate(reader):
                if len(row) != len(header):
                    raise SystemExit(f"{path}: row {k + 2} has {len(row)} fields")
                r = dict(zip(header, row))
                if r["src"] == "nl" and r["dest"] == "en":
                    if r["direct"] != "1":
                        raise SystemExit(f"{path}: {r['iid']} is not a direct translation")
                    out.append({"source": r["text"], "reference": r["original"],
                                "meta": {"iid": r["iid"], "split": split, "native_speaker": r["native_speaker"]}})
    iids = [r["meta"]["iid"] for r in out]
    if len(set(iids)) != len(iids):
        raise SystemExit("Europarl-direct: an iid occurs twice among the nl->en rows")
    return out


def length_class(tokens):
    for name in CLASS_ORDER:
        if tokens <= LENGTH_CLASSES[name]["max_source"]:
            return name
    raise SystemExit(f"a source of {tokens} tokens is longer than translate.py's longest class")


def class_seconds(rows):
    return LOAD_SECONDS + DUTCH_PER_ENGLISH * sum(r["tokens"] for r in rows) / OUT_PER_SECOND


def assign_workers(rows, workers):
    """{worker: [rows]}: whole length classes, largest estimated time first, each to the least loaded worker"""
    by_class = {}
    for r in rows:
        by_class.setdefault(r["length_class"], []).append(r)
    load = {k: 0.0 for k in range(workers)}
    out = {k: [] for k in range(workers)}
    for name in sorted(by_class, key=lambda c: (-class_seconds(by_class[c]), CLASS_ORDER.index(c))):
        k = min(load, key=lambda w: (load[w], w))
        load[k] += class_seconds(by_class[name])
        out[k] += by_class[name]
    return {k: sorted(v, key=lambda r: (r["tokens"], r["id"])) for k, v in out.items()}


def fetch(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    out = Path(args.out)
    dl = out / "download"
    wmt = checked(Path(hf_hub_download(WMT24PP[0], "en-nl_NL.jsonl", repo_type="dataset", revision=WMT24PP[1],
                                       local_dir=str(dl / "wmt24pp"))))
    archive = checked(download(FLORES_URL, dl / "flores" / "flores200_dataset.tar.gz"))
    flores = {}
    with tarfile.open(archive) as tar:
        for member in ("devtest/eng_Latn.devtest", "devtest/nld_Latn.devtest", "metadata_devtest.tsv"):
            f = tar.extractfile(f"./flores200_dataset/{member}")
            if f is None:
                raise SystemExit(f"{archive}: no {member}")
            path = dl / "flores" / Path(member).name
            path.write_bytes(f.read())
            flores[Path(member).name] = checked(path)
    ntrex = {name: checked(download(NTREX_URL.format(f"NTREX-128/{name}"), dl / "ntrex" / name))
             for name in ("newstest2019-src.eng.txt", "newstest2019-ref.nld.txt")}
    europarl = [(split, checked(download(ZENODO.format(f"multi8_{split}.tsv"), dl / "europarl" / f"multi8_{split}.tsv")))
                for split in ("train", "dev", "test")]

    sets = {"wmt24pp": read_wmt24pp(wmt),
            "flores": read_flores(flores["eng_Latn.devtest"], flores["nld_Latn.devtest"], flores["metadata_devtest.tsv"]),
            "ntrex": read_ntrex(ntrex["newstest2019-src.eng.txt"], ntrex["newstest2019-ref.nld.txt"]),
            "europarl": read_europarl(europarl)}
    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    rows = []
    for name, segments in sets.items():
        if len(segments) != SETS[name]["count"]:
            raise SystemExit(f"{name}: {len(segments)} segments, expected {SETS[name]['count']}")
        for k, s in enumerate(segments):
            if not s["source"].strip() or not s["reference"].strip():
                raise SystemExit(f"{name} {k}: an empty source or reference")
            n = len(tok.encode(s["source"], add_special_tokens=False).ids)
            rows.append({"id": f"{name}-{k:04d}", "set": name, "source": s["source"], "reference": s["reference"],
                         "tokens": n, "over_512": n > MARK_OVER, "length_class": length_class(n),
                         "meta": json.dumps(s["meta"], ensure_ascii=False)})
    columns = ["id", "set", "source", "reference", "tokens", "over_512", "length_class", "meta"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in columns}), out / "evalsets.parquet")
    assignment = assign_workers(rows, args.workers)
    for k, part in assignment.items():
        pq.write_table(pa.table({"id": pa.array([r["id"] for r in part], pa.string()),
                                 "text": pa.array([r["source"] for r in part], pa.string()),
                                 "tokens": pa.array([r["tokens"] for r in part], pa.int64())}),
                       out / f"segments-w{k}.parquet")
        classes = [c for c in CLASS_ORDER if any(r["length_class"] == c for r in part)]
        (out / f"classes-w{k}.txt").write_text(" ".join(classes) + "\n")

    files = sorted(p for p in dl.rglob("*") if p.is_file() and not p.name.endswith(".part") and ".cache" not in p.parts)
    manifest = {
        "sets": {name: {**SETS[name], "segments": sum(r["set"] == name for r in rows),
                        "source_tokens": sum(r["tokens"] for r in rows if r["set"] == name),
                        "max_tokens": max(r["tokens"] for r in rows if r["set"] == name),
                        "over_512": sum(r["over_512"] for r in rows if r["set"] == name),
                        "length_classes": {c: sum(r["length_class"] == c for r in rows if r["set"] == name)
                                           for c in CLASS_ORDER}}
                 for name in SETS},
        "files": {str(p.relative_to(out)): sha256(p) for p in files},
        "tokenizer": str(args.tokenizer),
        "workers": {f"w{k}": {"segments": len(part), "source_tokens": sum(r["tokens"] for r in part),
                              "classes": {c: sum(r["length_class"] == c for r in part) for c in CLASS_ORDER
                                          if any(r["length_class"] == c for r in part)},
                              "estimated_seconds": round(sum(class_seconds([r for r in part if r["length_class"] == c])
                                                             for c in {r["length_class"] for r in part}))}
                    for k, part in assignment.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for name, s in manifest["sets"].items():
        print(f"SET {name}: {s['segments']} segments, {s['source_tokens']:,} source tokens, max {s['max_tokens']}, "
              f"over 512: {s['over_512']}", flush=True)
    for w, s in manifest["workers"].items():
        print(f"WORKER {w}: {s['segments']} segments, classes {s['classes']}, ~{s['estimated_seconds']} s", flush=True)


def read_shards(root):
    """{id: row} of translate.py's shards under root (<class>/shard-*.parquet)"""
    import pyarrow.parquet as pq

    out = {}
    for path in sorted(Path(root).rglob("shard-*.parquet")):
        for r in pq.read_table(path).to_pylist():
            if r["id"] in out:
                raise SystemExit(f"{r['id']} is in two shards ({path})")
            out[r["id"]] = r
    return out


def collect(args):
    import publish
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = Path(args.data)
    sets = {r["id"]: r for r in pq.read_table(data / "evalsets.parquet").to_pylist()}
    mine = pq.read_table(data / f"segments-w{args.worker}.parquet", columns=["id"]).column("id").to_pylist()
    shards = read_shards(args.shards)
    missing, extra = [i for i in mine if i not in shards], sorted(set(shards) - set(mine))
    if missing or extra:
        raise SystemExit(f"worker {args.worker}: {len(missing)} segments without a translation ({missing[:5]}), "
                         f"{len(extra)} translations of segments of another worker ({extra[:5]})")
    rows = []
    for i in mine:
        s, t = sets[i], shards[i]
        hyp = t["text_nl"]
        rows.append({"id": i, "set": s["set"], "source": s["source"], "hypothesis": hyp, "reference": s["reference"],
                     "ended": t["ended"], "tokens_in": t["tokens_in"], "tokens_out": t["tokens_out"],
                     "decode_loop": bool(hyp) and publish.decode_loop(hyp, s["source"]),
                     "english_left": bool(hyp) and publish.english_left(hyp, s["source"]),
                     "empty": not hyp.strip(), "tokens": s["tokens"], "over_512": s["over_512"],
                     "length_class": s["length_class"]})
    columns = list(rows[0]) if rows else ["id"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in columns}), args.out)
    print(f"COLLECT w{args.worker}: {len(rows)} pairs, not ended {sum(not r['ended'] for r in rows)}, decode_loop "
          f"{sum(r['decode_loop'] for r in rows)}, english_left {sum(r['english_left'] for r in rows)}, empty "
          f"{sum(r['empty'] for r in rows)}", flush=True)


def stats(values):
    return {"mean": round(statistics.fmean(values), 4), "median": round(statistics.median(values), 4)} if values else None


def set_summary(rows):
    out = {"segments": len(rows),
           "metricx_ref": stats([r["metricx_ref"] for r in rows]),
           "metricx_qe": stats([r["metricx_qe"] for r in rows]),
           "comet22": stats([r["comet22"] for r in rows]),
           "not_ended": sum(not r["ended"] for r in rows),
           "decode_loop": sum(r["decode_loop"] for r in rows),
           "not_ended_or_loop": sum(not r["ended"] or r["decode_loop"] for r in rows),
           "english_left": sum(r["english_left"] for r in rows),
           "empty": sum(r["empty"] for r in rows),
           "over_512": sum(r["over_512"] for r in rows)}
    return out


def subsets(rows):
    """{label: rows}: all segments, and the subsets the summary also reports where they differ from all"""
    out = {"all": rows}
    good = [r for r in rows if not json.loads(r["meta"]).get("is_bad_source", False)]
    if len(good) != len(rows):
        out["without_bad_source"] = good  # WMT24++: the paper leaves is_bad_source out, the card recommends it
    short = [r for r in rows if not r["over_512"]]
    if len(short) != len(rows):
        out["up_to_512_tokens"] = short
    return out


def fmt(s, key):
    return "–" if s[key] is None else f"{s[key]['mean']:.3f} / {s[key]['median']:.3f}"


def summary_md(summary):
    lines_ = [f"# Uitgangspunt en→nl: {summary['model']}", "",
              f"Gemaakt {summary['created']} door `evalsets.py summary` (branch `en-nl-eval`). Ontwerp §4 "
              "(hammond `mbert/designs/en-nl-translator.md`). Per cel gemiddelde / mediaan. MetricX-24 hybrid XXL: "
              "lager is beter (0–25); COMET-22: hoger is beter. 'Niet af' = gestopt op de decodeerlimiet; "
              "'lus' = `decode_loop` van FLAN-nl; '> 512' = bron boven 512 tokens (gemarkeerd).", "",
              "| set | deel | n | MetricX ref | MetricX QE | COMET-22 | niet af | lus | niet af of lus | Engels over | "
              "leeg | > 512 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    explain = ["", "Deel: `all` = alle segmenten; `without_bad_source` = zonder de segmenten met `is_bad_source` "
               "(WMT24++; het artikel laat ze weg en de datasetkaart raadt dat aan); `up_to_512_tokens` = zonder de "
               "gemarkeerde bronnen boven 512 tokens.", "",
               "Beslisregel §4 (a): een arm mag op WMT24++, FLORES en NTREX met MetricX-24-QE niet meer dan 0,2 punt "
               "slechter zijn dan deze waarden. Europarl-direct wordt gerapporteerd, maar beslist niet."]
    for name, parts in summary["sets"].items():
        for label, s in parts.items():
            lines_.append(f"| {name} | {label} | {s['segments']} | {fmt(s, 'metricx_ref')} | {fmt(s, 'metricx_qe')} | "
                          f"{fmt(s, 'comet22')} | {s['not_ended']} | {s['decode_loop']} | {s['not_ended_or_loop']} | "
                          f"{s['english_left']} | {s['empty']} | {s['over_512']} |")
    lines_ += explain + ["", "## Opzet", ""]
    for key, value in summary["setup"].items():
        lines_.append(f"- **{key}**: {value}")
    lines_ += ["", "## Sets", ""]
    for name, s in summary["data"]["sets"].items():
        tokens = f"{s['source_tokens']:,}".replace(",", ".")
        lines_.append(f"- **{name}** ({s['name']}): {s['segments']} segmenten, {tokens} brontokens "
                      f"(max {s['max_tokens']}, boven 512: {s['over_512']}). Bron: {s['source']}. Licentie: "
                      f"{s['license']}. Referentie: {s['reference']}.")
    if summary.get("comet_timing"):
        lines_ += ["", "## Duur van COMET-22 (CPU)", ""]
        for w, t in sorted(summary["comet_timing"].items()):
            lines_.append(f"- {w}: eerste {t.get('first')} segmenten in {t.get('first_seconds')} s "
                          f"(projectie {t.get('projected_seconds')} s voor {t['rows']}), rest {t.get('rest_seconds')} s; "
                          f"{t['threads']} threads, batch {t['batch_size']}")
    return "\n".join(lines_) + "\n"


def summary(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = Path(args.data)
    sets = {r["id"]: r for r in pq.read_table(data / "evalsets.parquet").to_pylist()}
    manifest = json.loads((data / "manifest.json").read_text())

    def joined(paths, what):
        out = {}
        for p in paths:
            for r in pq.read_table(p).to_pylist():
                if r["id"] in out:
                    raise SystemExit(f"{r['id']} twice in the {what}")
                out[r["id"]] = r
        if set(out) != set(sets):
            raise SystemExit(f"the {what} have {len(out)} segments, the sets {len(sets)}: missing "
                             f"{sorted(set(sets) - set(out))[:5]}, extra {sorted(set(out) - set(sets))[:5]}")
        return out

    pairs, metricx, comet = joined(args.pairs, "pairs"), joined(args.metricx, "MetricX scores"), joined(args.comet,
                                                                                                    "COMET scores")
    rows = []
    for i in sets:  # the order of evalsets.parquet
        p, m, c = pairs[i], metricx[i], comet[i]
        r = {**p, "meta": sets[i]["meta"], "metricx_ref": m[args.metricx_ref_column],
             "metricx_qe": m[args.metricx_qe_column], "comet22": c["comet22"]}
        for key in ("metricx_ref", "metricx_qe", "comet22"):
            if r[key] is None or not math.isfinite(r[key]):
                raise SystemExit(f"{i}: no {key} score")
        rows.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text_columns = ["id", "set", "source", "reference", "hypothesis", "ended", "decode_loop", "english_left", "empty",
                    "tokens", "tokens_in", "tokens_out", "over_512", "length_class", "meta"]
    score_columns = ["id", "set", "metricx_ref", "metricx_qe", "comet22", "ended", "decode_loop", "english_left",
                     "empty", "over_512"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in text_columns}), out / "translations.parquet")
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in score_columns}), out / "scores.parquet")
    timing = {}
    for path in args.comet_timing or []:
        timing[Path(path).stem.replace("comet-timing-", "")] = json.loads(Path(path).read_text())
    result = {
        "model": args.model,
        "created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "sets": {name: {label: set_summary(part) for label, part in subsets([r for r in rows if r["set"] == name]).items()}
                 for name in SETS},
        "setup": json.loads(args.setup) if args.setup else {},
        "data": {"sets": manifest["sets"], "files": manifest["files"]},
        "comet_timing": timing,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text(summary_md(result))
    for name, parts in result["sets"].items():
        s = parts["all"]
        print(f"SUMMARY {name}: n {s['segments']}, MetricX ref {fmt(s, 'metricx_ref')}, QE {fmt(s, 'metricx_qe')}, "
              f"COMET-22 {fmt(s, 'comet22')}, not ended {s['not_ended']}, loop {s['decode_loop']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True, help="TranslateGemma's tokenizer directory (tokenizer.json)")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("collect")
    p.add_argument("--data", required=True, help="fetch's directory")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--shards", required=True, help="A local copy of the worker's shards (<class>/shard-*.parquet)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--data", required=True)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--metricx", nargs="+", required=True)
    p.add_argument("--comet", nargs="+", required=True)
    p.add_argument("--comet-timing", nargs="*", default=[])
    p.add_argument("--metricx-ref-column", default="ref_source_hypothesis_reference")
    p.add_argument("--metricx-qe-column", default="qe_source_hypothesis")
    p.add_argument("--model", required=True, help="The translator, as the summary names it")
    p.add_argument("--setup", default=None, help="JSON object: how the translations and scores were made")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"fetch": fetch, "collect": collect, "summary": summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
