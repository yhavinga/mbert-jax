#!/usr/bin/env python3
"""The pairs the judge scores: English-original Europarl paragraphs with their Dutch translation, and the sanity set.

    python data.py motra --repo <checkout of Multilingual-Parallel-Direct-Europarl> --out <dir>
    python data.py sanity --pairs <dir of motra> --out <dir> [--seed 0]
    python data.py pilot --pairs pairs.parquet --qe qe.parquet --out <dir>

pilot: the 2,000 pairs of the translator's pilot (mbert/translator/pilot/pairs.parquet with qe.parquet), scored as the
training pair of the fine-tune: the English source is the synthetic English (nl->en by TranslateGemma 27B), the Dutch
translation is the Dutch original. chrF (back-translation against the original) and MetricX-24-QE are carried along.
The id is the pair's pid (the passage id that pairs30.md shows); its numeric id, the key of qe.parquet, is row_id.

motra: Multilingual Parallel Direct Europarl (Amponsah-Kaakyire et al., MoTra 2021), proceedings/xml/{en,nl}, one
file per sitting. A paragraph with sl="en" in an intervention of the English file is paired with the Dutch file's
paragraph of the same intervention id, the same index among that intervention's sl="en" paragraphs and the same number
of such paragraphs: the key {date}:{intervention}:{index}:{count} of the repo's scripts/utils.py. Unlike utils.py
there is no langdetect filter (a translation that is not Dutch is for the judge to find) and the text is all of the
paragraph's text (itertext), not only p.text before a first child element. An intervention id that occurs twice in
one file pairs nothing (its keys are ambiguous; utils.py keeps the last one, its write_to_tsv drops duplicate keys).
From mid-2011 on this pairs nothing, as in utils.py: the Dutch files mark every paragraph sl="nl", and both files hold
the untranslated verbatim speeches (checked on 9 Nov on 2011-07-05, 2011-09-15, 2012-01-17 and 2012-09-12: paired by
position within an intervention, the "Dutch" side equalled the English side in 1,446 of 1,570 paragraphs, and the rest
were mostly other languages), so there is no translation to pair.
Output: shard-NNNNN.parquet in sitting order, manifest.json with the counts.

sanity: 500 paragraphs as they are and deterministic errors made from them (no text is written by anyone; every
corrupted Dutch side is the original Dutch with a rule applied, or another row's Dutch, or the English):
    mismatch          the Dutch of another of the 500 (a seeded derangement)
    sentence_dropped  one sentence of the Dutch left out (seeded choice), if the Dutch has two or more
    number_changed    the last digit of the first number in the Dutch that also occurs in the English, plus one mod 10
    negation_removed  the only "niet" removed or the only "geen" made "een", where the Dutch has exactly one of them
                      and no other negation word
    english_kept      the English as the Dutch side
The 500 are drawn in four strata of 125 (seeded, without overlap) so that every rule has enough rows: paragraphs where
a sentence can be dropped, where a number can be changed, where a negation can be removed, and any paragraph. Every
rule is then applied to every one of the 500 where it applies.
"""

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------- motra

SPEAKER_FIELDS = ("speaker_id", "name", "nationality", "mode", "role", "is_mep", "m_state")


def paragraph_text(p):
    return " ".join("".join(p.itertext()).split())


def sitting_paragraphs(path, sl="en"):
    """{key: (paragraph text, intervention attributes)} of the sl paragraphs of one sitting's file, and the number of
    keys left out because their intervention id occurs more than once in the file."""
    root = ET.parse(path).getroot()
    date = root.get("id").split(".")[0]
    seen = Counter(el.get("id") for el in root.iter("intervention"))
    out, ambiguous = {}, 0
    for el in root.iter("intervention"):
        iid = el.get("id")
        paragraphs = el.findall(f'.//p[@sl="{sl}"]')
        if seen[iid] > 1:
            ambiguous += len(paragraphs)
            continue
        attrs = {k: el.get(k) for k in SPEAKER_FIELDS}
        for i, p in enumerate(paragraphs):
            out[f"{date}:{iid}:{i}:{len(paragraphs)}"] = (paragraph_text(p), attrs, len(p) > 0)
    return root.get("date"), out, ambiguous


def pair_sitting(paths):
    en_path, nl_path = paths
    date, en, en_ambiguous = sitting_paragraphs(en_path)
    counts = Counter(en_paragraphs=len(en) + en_ambiguous, en_ambiguous=en_ambiguous)
    if nl_path is None:
        counts["no_nl_file"] = len(en)
        return date, [], counts
    _, nl, nl_ambiguous = sitting_paragraphs(nl_path)
    counts["nl_ambiguous_keys"] = nl_ambiguous
    rows = []
    for key, (en_text, attrs, en_children) in en.items():
        if key not in nl:
            counts["no_nl_paragraph"] += 1
            continue
        nl_text, _, nl_children = nl[key]
        counts["paired"] += 1
        counts["with_child_elements"] += int(en_children or nl_children)
        _, iid, index, count = key.split(":")
        rows.append({"id": key, "date": date, "year": int(date[:4]), "intervention_id": iid,
                     "paragraph": int(index), "paragraphs": int(count), "speaker": attrs["name"],
                     "speaker_id": attrs["speaker_id"], "nationality": attrs["nationality"],
                     "mode": attrs["mode"], "role": attrs["role"], "sl": "en", "en": en_text, "nl": nl_text,
                     "en_words": len(en_text.split()), "nl_words": len(nl_text.split())})
    return date, rows, counts


def motra_command(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    xml = Path(args.repo) / "proceedings" / "xml"
    en_files = sorted((xml / "en").glob("*.EN.xml"))
    if not en_files:
        raise SystemExit(f"no *.EN.xml in {xml / 'en'}")
    jobs = []
    for path in en_files:
        nl_path = xml / "nl" / path.name.replace(".EN.xml", ".NL.xml")
        jobs.append((path, nl_path if nl_path.exists() else None))
    with Pool(args.processes) as pool:
        results = pool.map(pair_sitting, jobs, chunksize=4)
    rows, totals, by_period, missing = [], Counter(), {}, {}
    for (path, _), (date, sitting_rows, counts) in zip(jobs, results):
        rows.extend(sitting_rows)
        totals.update(counts)
        period = "1999-2003" if int(path.name[:4]) <= 2003 else "2004-2012"
        by_period.setdefault(period, Counter()).update(counts)
        lost = counts["no_nl_paragraph"] + counts["no_nl_file"]
        if lost:
            missing[date] = lost
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    for k in range(0, len(rows), args.shard_size):
        chunk = rows[k:k + args.shard_size]
        pq.write_table(pa.table({c: [r[c] for r in chunk] for c in columns}),
                       out / f"shard-{k // args.shard_size:05d}.parquet", compression="zstd")
    manifest = {"repo_commit": args.commit, "sittings_en": len(en_files), "sittings_without_nl_file": sum(nl is None for _, nl in jobs),
                "counts": dict(totals), "by_period": {k: dict(v) for k, v in sorted(by_period.items())},
                "by_year": dict(sorted(Counter(r["year"] for r in rows).items())),
                "dates_with_unpaired_en": dict(sorted(missing.items())), "shard_size": args.shard_size,
                "shards": (len(rows) + args.shard_size - 1) // args.shard_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("repo_commit", "sittings_en", "sittings_without_nl_file", "counts", "by_period")},
                     indent=1))


# --------------------------------------------------------------------------------------------------------------- sanity

# The end of a sentence: . ! or ? with any closing quotes or parenthesis, white space, then a capital (after any opening
# quote); the closing quotes stay with the sentence they close
SENTENCE_BREAK = re.compile(r"[.!?][\"'’”)]*(\s+)(?=[\"'‘“(]*[A-ZÀ-Þ])")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NEGATION = re.compile(r"\b(niet|geen)\b", re.IGNORECASE)
OTHER_NEGATION = re.compile(r"\b(nooit|niets|niemand|nergens|noch|geenszins|nimmer|allerminst)\b", re.IGNORECASE)
KINDS = ("original", "mismatch", "sentence_dropped", "number_changed", "negation_removed", "english_kept")


def sentences(text):
    parts, start = [], 0
    for m in SENTENCE_BREAK.finditer(text):
        parts.append(text[start:m.start(1)])
        start = m.end(1)
    parts.append(text[start:])
    return [part for part in parts if part.strip()]


def drop_sentence(nl, rng):
    parts = sentences(nl)
    if len(parts) < 2:
        return None
    k = rng.randrange(len(parts))
    return " ".join(parts[:k] + parts[k + 1:])


def change_number(en, nl):
    en_numbers = {re.sub(r"\D", "", m) for m in NUMBER.findall(en)}
    for m in NUMBER.finditer(nl):
        if re.sub(r"\D", "", m.group()) in en_numbers:
            old = m.group()
            new = old[:-1] + str((int(old[-1]) + 1) % 10)
            return nl[:m.start()] + new + nl[m.end():]
    return None


def remove_negation(nl):
    found = list(NEGATION.finditer(nl))
    if len(found) != 1 or OTHER_NEGATION.search(nl):
        return None
    m = found[0]
    word = m.group()
    if word.lower() == "geen":
        return nl[:m.start()] + ("Een" if word[0] == "G" else "een") + nl[m.end():]
    # "niet" only between two spaces, so that removing it leaves a sentence with its punctuation intact
    if m.start() == 0 or nl[m.start() - 1] != " " or nl[m.end():m.end() + 1] != " ":
        return None
    return nl[:m.start() - 1] + nl[m.end():]


def corruptions(rows, seed):
    """The sanity rows: every original and every rule applied where it applies; ids are '<source id>|<kind>'."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    partner = {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}
    out = []
    for i, row in enumerate(rows):
        en, nl = row["en"], row["nl"]
        made = {"original": nl, "mismatch": rows[partner[i]]["nl"],
                "sentence_dropped": drop_sentence(nl, random.Random(f"{seed}:{row['id']}")),
                "number_changed": change_number(en, nl), "negation_removed": remove_negation(nl),
                "english_kept": en}
        for kind in KINDS:
            if made[kind] is not None:
                out.append({"id": f"{row['id']}|{kind}", "kind": kind, "source_id": row["id"], "en": en,
                            "nl": made[kind]})
    return out


def draw_strata(rows, seed, per_stratum=125):
    """Four disjoint seeded draws of per_stratum rows: sentence can be dropped, number can be changed, negation can be
    removed, and any row."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    tests = [lambda r: len(sentences(r["nl"])) >= 2,lambda r: change_number(r["en"], r["nl"]) is not None,
             lambda r: remove_negation(r["nl"]) is not None, lambda r: True]
    chosen, taken = [], set()
    for test in tests:
        picked = [i for i in order if i not in taken and test(rows[i])][:per_stratum]
        if len(picked) < per_stratum:
            raise SystemExit(f"only {len(picked)} rows for a stratum of {per_stratum}")
        taken.update(picked)
        chosen.extend(picked)
    return [rows[i] for i in sorted(chosen)]


def sanity_command(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables([pq.read_table(p) for p in sorted(Path(args.pairs).glob("shard-*.parquet"))])
    rows = [r for r in table.select(["id", "en", "nl", "en_words", "nl_words"]).to_pylist()
            if args.min_words <= r["en_words"] <= args.max_words and args.min_words <= r["nl_words"] <= args.max_words
            and r["en"] != r["nl"]]
    sample = draw_strata(rows, args.seed)
    out_rows = corruptions(sample, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: [r[c] for r in out_rows] for c in ("id", "kind", "source_id", "en", "nl")}),
                   out / "shard-00000.parquet", compression="zstd")
    counts = Counter(r["kind"] for r in out_rows)
    (out / "manifest.json").write_text(json.dumps({"pool": len(rows), "seed": args.seed, "kinds": counts}, indent=1))
    print(json.dumps({"pool": len(rows), "kinds": counts}))


PILOT_KEEP = ("source", "half", "chrf", "tokens_nl", "tokens_en")


def pilot_command(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.pairs)
    qe = pq.read_table(args.qe)
    print("pairs.parquet:", table.schema.to_string(show_schema_metadata=False).replace("\n", "; "))
    print("qe.parquet:", qe.schema.to_string(show_schema_metadata=False).replace("\n", "; "))
    need = {"id", "pid", "nl", "en", "chrf"}
    if not need <= set(table.column_names) or not {"id", "qe_en_nl", "qe_nl_en"} <= set(qe.column_names):
        raise SystemExit(f"pairs.parquet needs {sorted(need)}, qe.parquet id, qe_en_nl, qe_nl_en")
    rows = table.to_pylist()
    scores = {r["id"]: r for r in qe.to_pylist()}
    if len(scores) != qe.num_rows or len({r["id"] for r in rows}) != len(rows) or \
            len({r["pid"] for r in rows}) != len(rows):
        raise SystemExit("duplicate ids")
    missing = [r["id"] for r in rows if r["id"] not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} pairs without QE, e.g. {missing[:3]}")
    out = {"id": [r["pid"] for r in rows], "row_id": [r["id"] for r in rows], "en": [r["en"] for r in rows],
           "nl": [r["nl"] for r in rows]}
    for c in PILOT_KEEP:
        if c in table.column_names:
            out[c] = [r[c] for r in rows]
    out["qe_en_nl"] = [scores[r["id"]]["qe_en_nl"] for r in rows]
    out["qe_nl_en"] = [scores[r["id"]]["qe_nl_en"] for r in rows]
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), path / "shard-00000.parquet", compression="zstd")
    (path / "manifest.json").write_text(json.dumps({"pairs": len(rows), "columns": list(out),
                                                    "source_columns": table.column_names}, indent=1))
    print(f"{len(rows)} pilot pairs -> {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pairs for the en-nl judge")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("motra")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard-size", type=int, default=10000)
    p.add_argument("--processes", type=int, default=32)
    p.add_argument("--commit", default=None, help="The commit of the repo checkout, for the manifest")
    p = sub.add_parser("sanity")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--max-words", type=int, default=150)
    p = sub.add_parser("pilot")
    p.add_argument("--pairs", required=True)
    p.add_argument("--qe", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"motra": motra_command, "sanity": sanity_command, "pilot": pilot_command}[args.command](args)


if __name__ == "__main__":
    main()
