#!/usr/bin/env python3
"""Download the evaluation sets of experiments/dumb-extra from pinned sources and write them as parquet with a manifest.

Usage (on the Mac, once; the VM jobs read the result from the bucket):
    python -I experiments/dumb-extra/stage_data.py --download-dir DL --output-dir OUT
    gsutil -m cp -r OUT/* gs://t5-dutch-english-usc2/mbert/dumb-extra/data/

Every source is pinned: a DOI version, a Hugging Face revision, a git commit (of a release tag), a Wayback Machine
timestamp. BLiMP-NL's files are checked against the SHA256 sums its collection publishes (MANIFEST.txt); the CoNLL-2002
copy is checked against the archived official files. The sha256 of every downloaded file goes into manifest.json. A
failed check stops the run; nothing falls back to another copy.

These sets are evaluation data only and must never be part of pretraining data.
"""

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BLIMP_URL = "https://webdav.data.ru.nl/cls/blimp-nl_dsc_550_v1"
SQUAD_REVISION = "7e1167c6ba074b3aedcd11061b0a921d5383106d"
SQUAD_URL = f"https://huggingface.co/datasets/GroNLP/squad-nl-v1.1/resolve/{SQUAD_REVISION}"
# The copy that the conll2002 script of eriktks/conll2002 on Hugging Face reads (at master); pinned to its commit
CONLL_COMMIT = "f89bc5a15b516208de1cfe2abe9692b7ed9a43e2"
CONLL_URL = f"https://raw.githubusercontent.com/teropa/nlp/{CONLL_COMMIT}/resources/corpora/conll2002"
# The official files (www.clips.uantwerpen.be/conll2002/ner/data/, offline since), as archived by the Wayback Machine
CONLL_OFFICIAL = {
    "000README": "20201031160612", "ned.train": "20201031160719", "ned.testa": "20201031165756", "ned.testb": "20201031170055",
}
UD_TAG = "r2.18"
UD_TREEBANKS = {
    "ud-nl-lassysmall": ("UD_Dutch-LassySmall", "nl_lassysmall", "b8cdca9661408608945133e80bb566a6a58146ea",
                         " The UD treebank of the Wikipedia section of Lassy Small, not the IvdNT Lassy Small corpus of "
                         "DUMB's lassy-pos task."),
    "ud-nl-alpino": ("UD_Dutch-Alpino", "nl_alpino", "d64749ac918f18d31dd4c4e75ff56626432d9ba3", ""),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mbert-jax stage_data.py"})
    with urllib.request.urlopen(request, timeout=300) as response, open(path, "wb") as f:
        f.write(response.read())
    return path


def checked(path, expected, source):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual}, expected {expected} ({source}); the source changed")
    return actual


def write_parquet(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return {"file": path.name, "rows": len(rows), "sha256": sha256(path), "bytes": path.stat().st_size}


def stage_blimp(download_dir, output_dir):
    """The large version, one row per minimal pair; Item_nr 1-840 are the hand-crafted pairs of the small version."""
    root = download_dir / "blimp-nl"
    manifest_path = download(f"{BLIMP_URL}/MANIFEST.txt", root / "MANIFEST.txt")
    published = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(" ", 1)
        published[name] = digest
    names = ["ABOUT.txt", "LICENSE.txt", "README.txt", "BLiMP-NL/BLiMP-NL (small)/BLiMP-NL_small_stimuli.tsv"]
    names += [f"BLiMP-NL/BLiMP-NL (large)/Phenomenon{i:02d}.tsv" for i in range(1, 23)]
    sources = {}
    for name in names:
        path = download(f"{BLIMP_URL}/{urllib.request.quote(name)}", root / name)
        sources[name] = checked(path, published[name], "MANIFEST.txt of the collection")

    def read(path):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    rows = [row for i in range(1, 23) for row in read(root / f"BLiMP-NL/BLiMP-NL (large)/Phenomenon{i:02d}.tsv")]
    small = {(row["Item_nr"], row["Grammaticality"]): row["Sentence"] for row in read(root / "BLiMP-NL/BLiMP-NL (small)/BLiMP-NL_small_stimuli.tsv")}
    by_item = {}
    for row in rows:
        by_item.setdefault(int(row["Item_nr"]), {})[row["Grammaticality"]] = row
    pairs = []
    edited = 0
    for item in sorted(by_item):
        both = by_item[item]
        if sorted(both) != ["Grammatical", "Ungrammatical"]:
            raise ValueError(f"BLiMP-NL item {item}: conditions {sorted(both)}")
        g, u = both["Grammatical"], both["Ungrammatical"]
        for key in ("Phenomenon_nr", "Phenomenon_name", "Paradigm_nr", "Paradigm_name"):
            if g[key] != u[key]:
                raise ValueError(f"BLiMP-NL item {item}: {key} differs between the two sentences")
        in_small = (str(item), "Grammatical") in small
        if in_small:
            edited += sum(small[(str(item), c)] != both[c]["Sentence"] for c in ("Grammatical", "Ungrammatical"))
        pairs.append({
            "item_nr": item, "phenomenon_nr": int(g["Phenomenon_nr"]), "phenomenon": g["Phenomenon_name"],
            "paradigm_nr": int(g["Paradigm_nr"]), "paradigm": g["Paradigm_name"], "handcrafted": in_small,
            "grammatical": g["Sentence"], "ungrammatical": u["Sentence"],
            "critical_word_grammatical": g["Critical_word"], "critical_word_ungrammatical": u["Critical_word"],
            "cue_word_grammatical": g["Cue_word"], "cue_word_ungrammatical": u["Cue_word"],
        })
    paradigms = Counter(p["paradigm_nr"] for p in pairs)
    if len(pairs) != 8400 or len(paradigms) != 84 or set(paradigms.values()) != {100} or len({p["phenomenon_nr"] for p in pairs}) != 22:
        raise ValueError(f"BLiMP-NL: {len(pairs)} pairs, {len(paradigms)} paradigms, sizes {set(paradigms.values())}")
    if sum(p["handcrafted"] for p in pairs) != 840 or any(p["handcrafted"] != (p["item_nr"] <= 840) for p in pairs):
        raise ValueError("BLiMP-NL: the items of the small version are not Item_nr 1-840 of the large one")
    split = write_parquet(pairs, output_dir / "blimp-nl" / "large.parquet")
    return {
        "name": "BLiMP-NL (large)",
        "source": "https://doi.org/10.34973/tj4p-y007 (Radboud Data Repository, collection blimp-nl_dsc_550, version 1), "
                  f"files from {BLIMP_URL}",
        "revision": "DOI 10.34973/tj4p-y007, version 1 (dateModified 2025-03-07)",
        "license": "CC BY-SA 4.0. README.txt: 'the data is freely accessible under a CC-BY-SA-4.0 license'; "
                   "LICENSE.txt: the text of Attribution-ShareAlike 4.0 International. ABOUT.txt asks to acknowledge: "
                   "'Data were provided (in part) by the Radboud University, Nijmegen, The Netherlands'.",
        "citation": "Suijkerbuijk, Prins, de Heer Kloots, Zuidema, Frank (2025). BLiMP-NL: A Corpus of Dutch Minimal Pairs and "
                    "Acceptability Judgments for Language Model Evaluation. Computational Linguistics 51(4). "
                    "https://aclanthology.org/2025.cl-4.6",
        "notes": "One row per minimal pair (8,400; 84 paradigms of 100; 22 phenomena). handcrafted = Item_nr 1-840, the "
                 "pairs of the small version; the other 7,560 were generated with ChatGPT (GPT-3.5 Turbo) from them and "
                 f"checked by hand (the paper, sections 3.3 and 8.3). {edited} sentences of the hand-crafted items differ "
                 "between the small and the large file; the large file's text is staged.",
        "source_sha256": sources,
        "splits": {"large": split},
    }


def stage_squad(download_dir, output_dir):
    root = download_dir / "squad-nl-v1.1"
    sources = {}
    splits = {}
    readme = download(f"{SQUAD_URL}/README.md", root / "README.md")
    sources["README.md"] = sha256(readme)
    for split in ("train", "dev", "test"):
        name = f"data/{split}-00000-of-00001.parquet"
        path = download(f"{SQUAD_URL}/{name}", root / name)
        sources[name] = sha256(path)
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            answers = row["answers"]
            if not answers["text"]:
                raise ValueError(f"SQuAD-NL v1.1 {split} {row['id']}: no answer")
            for start, text in zip(answers["answer_start"], answers["text"]):
                if row["context"][start:start + len(text)] != text:
                    raise ValueError(f"SQuAD-NL v1.1 {split} {row['id']}: answer {text!r} not at {start}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"SQuAD-NL v1.1 {split}: ids not unique")
        splits[split] = write_parquet(rows, output_dir / "squad-nl-v1.1" / f"{split}.parquet")
    return {
        "name": "SQuAD-NL v1.1",
        "source": "https://huggingface.co/datasets/GroNLP/squad-nl-v1.1",
        "revision": SQUAD_REVISION,
        "license": "cc-by-sa-4.0 (the dataset card's metadata)",
        "notes": "Answerable questions only. train: SQuAD v1.1 train, Google Translate (February 2023); dev: SQuAD v1.1 dev "
                 "minus the XQuAD documents, Google Translate; test: the XQuAD documents, Google Translate and post-edited "
                 "by hand (the card). DUMB's squadnl task uses SQuAD-NL v2.0 (with unanswerable questions; "
                 "hf_datasets/squad-nl config nl-v2.0 in github.com/wietsedv/dumb at 2c0eba5), not this v1.1. Every "
                 "answer was checked to be at its answer_start.",
        "source_sha256": sources,
        "splits": splits,
    }


def read_conll2002(path):
    """Sentences of a Dutch CoNLL-2002 file as the conll2002 script of Hugging Face reads it: -DOCSTART- lines skipped."""
    sentences = []
    document = -1
    tokens, pos, ner = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-DOCSTART-") or line.strip() == "":
            if tokens:
                sentences.append({"document": document, "tokens": tokens, "pos_tags": pos, "ner_tags": ner})
                tokens, pos, ner = [], [], []
            if line.startswith("-DOCSTART-"):
                document += 1
            continue
        parts = line.split(" ")
        if len(parts) != 3:
            raise ValueError(f"{path}: {line!r} is not 'word pos tag'")
        tokens.append(parts[0])
        pos.append(parts[1])
        ner.append(parts[2])
    if tokens:
        sentences.append({"document": document, "tokens": tokens, "pos_tags": pos, "ner_tags": ner})
    for i, sentence in enumerate(sentences):
        sentence["id"] = i
    return sentences


def stage_conll2002(download_dir, output_dir):
    root = download_dir / "conll2002-nl"
    sources = {}
    for name, timestamp in CONLL_OFFICIAL.items():
        url = f"https://web.archive.org/web/{timestamp}id_/https://www.clips.uantwerpen.be/conll2002/ner/data/{name}"
        sources[f"official/{name}"] = sha256(download(url, root / "official" / name))
    splits = {}
    for split, name in (("train", "ned.train"), ("validation", "ned.testa"), ("test", "ned.testb")):
        path = download(f"{CONLL_URL}/{name}", root / "mirror" / name)
        sources[f"mirror/{name}"] = sha256(path)
        mirror = path.read_bytes()
        official = (root / "official" / name).read_bytes().decode("latin-1").encode("utf-8")
        # The archived ned.train is cut at 1 MiB (the capture); the archived test files are whole
        if name == "ned.train":
            if not mirror.startswith(official) or len(official) < 1_000_000:
                raise ValueError(f"CoNLL-2002 {name}: the mirror does not start with the archived official file")
        elif mirror != official:
            raise ValueError(f"CoNLL-2002 {name}: the mirror differs from the archived official file")
        rows = read_conll2002(path)
        tags = Counter(tag for row in rows for tag in row["ner_tags"])
        if not set(tags) <= {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"}:
            raise ValueError(f"CoNLL-2002 {name}: tags {sorted(tags)}")
        splits[split] = write_parquet(rows, output_dir / "conll2002-nl" / f"{split}.parquet")
        splits[split]["words"] = sum(len(row["tokens"]) for row in rows)
    readme = (root / "official" / "000README").read_text(encoding="latin-1")
    terms = readme[readme.index("* Files in these"):readme.index("* Some redundant")].strip()
    return {
        "name": "CoNLL-2002 NER, Dutch",
        "source": "https://www.clips.uantwerpen.be/conll2002/ner/ (offline; data/ archived by the Wayback Machine on "
                  "2020-10-31); the parquet is made from the copy at github.com/teropa/nlp (resources/corpora/conll2002), "
                  "which the conll2002 script of huggingface.co/datasets/eriktks/conll2002 reads",
        "revision": f"teropa/nlp commit {CONLL_COMMIT}; checked equal to the archived official ned.testa and ned.testb "
                    "(latin-1 there, utf-8 here) and to the archived first MiB of ned.train",
        "license": "No license is stated. The official data/000README: '" + " ".join(terms.split()) + "' The Hugging Face card: "
                   "'The licensing status of the data, especially the news source text, is unknown.'",
        "notes": "train = ned.train, validation = ned.testa, test = ned.testb. IOB2 tags; the POS tags are MBT's unchecked "
                 "ones. -DOCSTART- lines are skipped, as in the Hugging Face script; 'document' counts them. The README "
                 "also says the sentences were randomized within each article for copyright reasons.",
        "source_sha256": sources,
        "splits": splits,
    }


def read_conllu(path):
    """Sentences of a CoNLL-U file: the syntactic words (integer ids), their UPOS; ranges and empty nodes left out."""
    sentences = []
    comments, tokens, upos = {}, [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            if tokens:
                sentences.append({"sent_id": comments.get("sent_id", ""), "text": comments.get("text", ""), "tokens": tokens, "upos": upos})
            comments, tokens, upos = {}, [], []
        elif line.startswith("#"):
            key, _, value = line[1:].partition("=")
            comments[key.strip()] = value.strip()
        else:
            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"{path}: {line!r} has {len(columns)} columns")
            if "-" in columns[0] or "." in columns[0]:
                continue
            tokens.append(columns[1])
            upos.append(columns[3])
    if tokens:
        sentences.append({"sent_id": comments.get("sent_id", ""), "text": comments.get("text", ""), "tokens": tokens, "upos": upos})
    return sentences


def stage_ud(download_dir, output_dir, name):
    repository, prefix, commit, note = UD_TREEBANKS[name]
    root = download_dir / name
    base = f"https://raw.githubusercontent.com/UniversalDependencies/{repository}/{commit}"
    sources = {}
    for file in ("LICENSE.txt", "README.txt"):
        sources[file] = sha256(download(f"{base}/{file}", root / file))
    splits = {}
    for split in ("train", "dev", "test"):
        file = f"{prefix}-ud-{split}.conllu"
        path = download(f"{base}/{file}", root / file)
        sources[file] = sha256(path)
        rows = read_conllu(path)
        splits[split] = write_parquet(rows, output_dir / name / f"{split}.parquet")
        splits[split]["words"] = sum(len(row["tokens"]) for row in rows)
    readme = (root / "README.txt").read_text(encoding="utf-8")
    license_line = next(line for line in readme.splitlines() if line.startswith("License:"))
    return {
        "name": repository,
        "source": f"https://github.com/UniversalDependencies/{repository}",
        "revision": f"tag {UD_TAG}, commit {commit}",
        "license": f"README.txt: '{license_line}'; LICENSE.txt: "
                   + " ".join((root / "LICENSE.txt").read_text(encoding="utf-8").split()[:24]) + " ...",
        "notes": "UPOS of the syntactic words (integer ids of CoNLL-U); multiword-token ranges and empty nodes left out."
                 + note,
        "source_sha256": sources,
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download-dir", required=True, type=Path, help="An empty directory for the downloads")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where the parquet files and manifest.json go")
    args = parser.parse_args()
    if args.download_dir.exists() and any(args.download_dir.iterdir()):
        raise SystemExit(f"--download-dir {args.download_dir} is not empty")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"--output-dir {args.output_dir} is not empty")
    datasets = {
        "blimp-nl": stage_blimp(args.download_dir, args.output_dir),
        "squad-nl-v1.1": stage_squad(args.download_dir, args.output_dir),
        "conll2002-nl": stage_conll2002(args.download_dir, args.output_dir),
        **{name: stage_ud(args.download_dir, args.output_dir, name) for name in UD_TREEBANKS},
    }
    manifest = {
        "staged": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/dumb-extra/stage_data.py (github.com/yhavinga/mbert-jax, branch dumb-extra)",
        "use": "Evaluation data only: never part of pretraining data.",
        "datasets": datasets,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for name, dataset in datasets.items():
        print(name, {split: values["rows"] for split, values in dataset["splits"].items()})


if __name__ == "__main__":
    main()
