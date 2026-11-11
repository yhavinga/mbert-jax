#!/usr/bin/env python3
"""The real translation pairs of arm R2 (protocol.md §9, amendment of 11 Nov: English-original text with a human Dutch
translation, in TranslateGemma's own prompt): Global Voices en->nl and Europarl-direct en->nl after 2003.

    python real_pairs.py gv-fetch --out DIR                 (the internet: nl.globalvoices.org and globalvoices.org)
    python real_pairs.py gv-align --fetched DIR --out gv.parquet
    python real_pairs.py europarl --motra DIR --out europarl.parquet       (DIR: data.py motra of en-nl-judge)
    python real_pairs.py candidates --gv gv.parquet --europarl europarl.parquet --blind blind100.jsonl \\
        --idioms idiom30.jsonl --out DIR                    (real.parquet, metricx-in-w<k>.parquet, candidates.json)
    python real_pairs.py join-qe --real DIR/real.parquet --qe metricx-w0.parquet ... --out real-qe.parquet

gv-fetch: every post of Global Voices in Dutch (WordPress API), and per post its page, where the list "Lees dit artikel
in het" names the English version; then the English version's page (is it itself a translation: the line
"Oorspronkelijk geplaatst"/post-source-date of a translated post; its author and the CC BY 3.0 licence of its footer)
and its text (WordPress API by slug). Raw JSONL in DIR, fetched once (a rerun fetches what is missing).
gv-align: per article whose English version is an original (no post-source-date on its page), is not a republished
partner text (blind.REPUBLISHED, as the blind passages) and carries CC BY 3.0 on both pages, the own paragraphs of both
sides (blind.paragraphs: not quotes, captions, embeds, tables, lists, italic notes), aligned in order (align): a
monotone alignment that pairs a paragraph with at most one paragraph of the other side, by length, shared links,
shared numbers and position; unpaired paragraphs (translator's notes, skipped paragraphs) stay out; the markers of a
linked page's language ("[en]", "[en – alle links]") are taken out of both sides.
europarl: the en->nl paragraphs of data.py motra (sl="en" in the English file, paired by key) from FROM_YEAR on; the
evaluation's Europarl-direct is the 640 nl->en paragraphs of 1999-2003 (design §6: train after 2003, by period).
candidates: the hard checks of a real pair (real_fails): too short, Dutch equal to the English, English left in the
Dutch (publish.english_left), length ratio in characters > LEN_RATIO_MAX (as pairs.py); and the English source of an
article of the blind assessment or the idiom set, or of the third subagent's pool (Global Voices posts of 2025) out.
The rest of the query (MetricX-QE, the bound on the length ratio in words, the 13-grams of the evaluation sets and the
blind passages, the tokens) is sft_build.py's, as for the back-translation pairs.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE.parents[1] / "experiments" / "en-nl-judge",
              HERE):
    sys.path.insert(0, str(_path))

GV_NL_API = "https://nl.globalvoices.org/wp-json/wp/v2/posts"
GV_EN_API = "https://globalvoices.org/wp-json/wp/v2/posts"
EN_LINK = re.compile(r"class='translation-language post-translation-en'><a href='(https://globalvoices\.org/[^']+)'")
SOURCE_DATE = re.compile(r"class='post-date post-source-date'>([^<]*)<")
LICENCE = "creativecommons.org/licenses/by/3.0"
FROM_YEAR = 2004       # Europarl: training after the evaluation's period (1999-2003), design §6
POOL_YEAR = "2025"     # the third subagent's pool of English posts (blind passages): left out whole
MIN_WORDS = 5          # English words of a pair, at least
LEN_RATIO_MAX = 1.5    # longer over shorter in characters, as pairs.py's hard check
SLUG_BATCH = 50
WORKERS = 4


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    # split on "\n" only: str.splitlines also splits at U+2028 and the like, which a JSON string may hold unescaped
    return [json.loads(line) for line in p.read_text(encoding="utf-8").split("\n") if line.strip()]


# ---------------------------------------------------------------------------------------------------- fetch

def gv_fetch(args):
    import blind

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nl_path = out / "nl-posts.jsonl"
    if not nl_path.exists():
        fields = "id,date,link,title,content,modified"
        page, total, rows = 1, None, []
        while total is None or page <= total:
            body, headers = blind.http_get(f"{GV_NL_API}?per_page=100&page={page}&orderby=date&order=asc&_fields={fields}")
            total = int(headers.get("X-WP-TotalPages") or headers.get("x-wp-totalpages"))
            rows += json.loads(body)
            page += 1
            time.sleep(args.sleep)
        tmp = nl_path.with_suffix(".part")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.rename(nl_path)
        print(f"FETCH nl posts {len(rows)} ({total} API pages)", flush=True)
    nl_posts = read_jsonl(nl_path)

    def page_facts(link):
        html, _ = blind.http_get(link)
        time.sleep(args.sleep)
        m = re.search(r'<meta name="author" content="([^"]+)"', html)
        return {"link": link, "en_link": (EN_LINK.search(html) or [None, None])[1],
                "source_date": (SOURCE_DATE.search(html) or [None, None])[1],
                "author": m.group(1) if m else None,
                "licence": LICENCE in html and "Creative Commons Attribution 3.0" in html,
                "bytes": len(html.encode("utf-8"))}

    def fetch_pages(links, path):
        done = {r["link"] for r in read_jsonl(path)}
        todo = [x for x in dict.fromkeys(links) if x not in done]
        with open(path, "a", encoding="utf-8") as f, \
                concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
            for k, facts in enumerate(pool.map(page_facts, todo)):
                f.write(json.dumps(facts, ensure_ascii=False) + "\n")
                f.flush()
                if k % 200 == 0:
                    print(f"PROGRESS {path.name} {k + 1}/{len(todo)}", flush=True)
        return read_jsonl(path)

    nl_pages = fetch_pages([p["link"] for p in nl_posts], out / "nl-pages.jsonl")
    print(f"FETCH nl pages {len(nl_pages)}, with an English version {sum(bool(p['en_link']) for p in nl_pages)}", flush=True)
    en_links = [p["en_link"] for p in nl_pages if p["en_link"]]
    en_pages = fetch_pages(en_links, out / "en-pages.jsonl")
    print(f"FETCH en pages {len(en_pages)}, translations themselves {sum(bool(p['source_date']) for p in en_pages)}",
          flush=True)

    en_path = out / "en-posts.jsonl"
    have = {r["link"] for r in read_jsonl(en_path)}
    slugs = {}
    for link in dict.fromkeys(en_links):
        if link not in have:
            slugs[link.rstrip("/").rsplit("/", 1)[-1]] = link
    fields = "id,date,link,title,content,modified,slug"
    items = sorted(slugs)
    with open(en_path, "a", encoding="utf-8") as f:
        for k in range(0, len(items), SLUG_BATCH):
            batch = items[k:k + SLUG_BATCH]
            body, _ = blind.http_get(f"{GV_EN_API}?per_page=100&slug={','.join(batch)}&_fields={fields}")
            time.sleep(args.sleep)
            for r in json.loads(body):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    en_posts = read_jsonl(en_path)
    found = {r["link"] for r in en_posts}
    missing = [x for x in dict.fromkeys(en_links) if x not in found]
    manifest = {"fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nl_api": GV_NL_API, "en_api": GV_EN_API,
                "nl_posts": len(nl_posts), "nl_pages": len(nl_pages), "en_links": len(set(en_links)),
                "en_pages": len(en_pages), "en_posts": len(en_posts), "en_links_without_api_post": len(missing),
                "sha256": {p.name: sha256_text(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.jsonl"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("FETCHED " + json.dumps({k: v for k, v in manifest.items() if k != "sha256"}), flush=True)


# ---------------------------------------------------------------------------------------------------- align

NUMBER = re.compile(r"\d+")
EXPECTED_CHAR_RATIO = 1.12  # Dutch over English characters of a human translation (Europarl after 2003: median 1.15)
# The Dutch counterparts of blind.BOILERPLATE: notes, credits and pointers that are no translation of a paragraph
NL_BOILERPLATE = re.compile(r"^\W*(?:noot van de (?:vertaler|redactie)|n\.?v\.?d\.?v|lees (?:ook|meer)|foto|afbeelding|"
                            r"schermafbeelding|bron:|vertaald|vertaling|dit (?:artikel|bericht|verhaal)|bekijk|luister|"
                            r"zie ook|gerelateerd|volg )", re.I)


# Global Voices marks the language of a linked page after the link: "[en]", "[es, fr]", "[EN]", "[en – alle links]",
# "[Engels – alle links]" (the Dutch side: 1,979 of 13,604 aligned pairs, 11 Nov; the English: 197). Metadata, no text
# of the paragraph: a Dutch side with "[en]" would teach the model to add it.
LANG_CODE = r"(?:[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?|Engels|Spaans|Frans|Arabisch|Russisch|Portugees|English|Spanish|French)"
LANG_MARK = re.compile(r"\s*\[-?" + LANG_CODE + r"(?:\s*[,/]\s*" + LANG_CODE + r")*(?:\s*[–—,-]\s*(?:alle|all)\s+links?)?\]")


def strip_lang_marks(text):
    return " ".join(LANG_MARK.sub("", text).split())


def norm_link(href):
    href = href.strip().split("#")[0].split("?")[0].rstrip("/").lower()
    return re.sub(r"^https?://(www\.)?", "", href)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def pair_score(en, nl, i, j, m, k):
    """How well English paragraph i of m and Dutch paragraph j of k match: their length against a human translation's,
    the links and numbers they share, and their place in the article. Above 0 they may be paired."""
    (en_text, en_links), (nl_text, nl_links) = en, nl
    score = 1.0 - 2.0 * abs(math.log(max(1, len(nl_text)) / max(1, len(en_text)) / EXPECTED_CHAR_RATIO))
    links_e, links_n = {norm_link(x) for x in en_links}, {norm_link(x) for x in nl_links}
    if links_e or links_n:
        score += 1.5 * jaccard(links_e, links_n) - 0.5
    nums_e, nums_n = set(NUMBER.findall(en_text)), set(NUMBER.findall(nl_text))
    if nums_e or nums_n:
        score += jaccard(nums_e, nums_n) - 0.5
    score -= 2.0 * abs(i / max(1, m) - j / max(1, k))
    return score


def align(en, nl):
    """[(i, j, score)]: a monotone alignment of the paragraphs en and nl ((text, links) each) that pairs a paragraph with
    at most one of the other side and maximizes the sum of the pairs' scores (pair_score), only scores above 0"""
    m, k = len(en), len(nl)
    best = [[0.0] * (k + 1) for _ in range(m + 1)]
    move = [[None] * (k + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            options = [(best[i - 1][j], "en"), (best[i][j - 1], "nl")]
            s = pair_score(en[i - 1], nl[j - 1], i - 1, j - 1, m, k)
            if s > 0:
                options.append((best[i - 1][j - 1] + s, "pair"))
            best[i][j], move[i][j] = max(options, key=lambda x: x[0])
    out, i, j = [], m, k
    while i > 0 and j > 0:
        step = move[i][j]
        if step == "pair":
            out.append((i - 1, j - 1, pair_score(en[i - 1], nl[j - 1], i - 1, j - 1, m, k)))
            i, j = i - 1, j - 1
        elif step == "en":
            i -= 1
        else:
            j -= 1
    return out[::-1]


def slug_of(link):
    return link.rstrip("/").rsplit("/", 1)[-1]


def gv_align(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import blind

    d = Path(args.fetched)
    nl_posts = {p["link"]: p for p in read_jsonl(d / "nl-posts.jsonl")}
    nl_pages = {p["link"]: p for p in read_jsonl(d / "nl-pages.jsonl")}
    en_pages = {p["link"]: p for p in read_jsonl(d / "en-pages.jsonl")}
    en_posts = {slug_of(p["link"]): p for p in read_jsonl(d / "en-posts.jsonl")}
    stats, rows, seen_en = collections.Counter(), [], set()
    for link, post in nl_posts.items():
        page = nl_pages.get(link)
        if page is None:
            raise SystemExit(f"{link}: no page fetched (gv-fetch first)")
        if not page["en_link"]:
            stats["no_english_version"] += 1
            continue
        en_page, en_post = en_pages.get(page["en_link"]), en_posts.get(slug_of(page["en_link"]))
        if en_page is None:
            raise SystemExit(f"{page['en_link']}: no page fetched (gv-fetch first)")
        if en_post is None:
            stats["english_not_in_api"] += 1
            continue
        if en_page["source_date"]:
            stats["english_is_a_translation"] += 1
            continue
        if not (en_page["licence"] and page["licence"]):
            stats["no_cc_by_3"] += 1
            continue
        en_html, nl_html = en_post["content"]["rendered"], post["content"]["rendered"]
        if blind.REPUBLISHED.search(re.sub(r"<[^>]+>", " ", en_html)):
            stats["republished"] += 1
            continue
        if en_post["id"] in seen_en:
            stats["second_translation_of_an_article"] += 1
            continue
        seen_en.add(en_post["id"])
        stats["articles"] += 1
        en_par, nl_par = blind.paragraphs(en_html, links=True), blind.paragraphs(nl_html, links=True)
        stats["paragraphs_en"] += len(en_par)
        stats["paragraphs_nl"] += len(nl_par)
        for i, j, score in align(en_par, nl_par):
            en_text, nl_text = strip_lang_marks(en_par[i][0]), strip_lang_marks(nl_par[j][0])
            stats["lang_marks_en"] += en_text != " ".join(en_par[i][0].split())
            stats["lang_marks_nl"] += nl_text != " ".join(nl_par[j][0].split())
            if blind.BOILERPLATE.match(en_text) or NL_BOILERPLATE.match(nl_text):
                stats["pairs_boilerplate"] += 1
                continue
            stats["pairs"] += 1
            rows.append({"id": f"gv:{en_post['id']}:{post['id']}:{i}:{j}", "source": "globalvoices", "en": en_text,
                         "nl": nl_text, "date": en_post["date"], "year": int(en_post["date"][:4]),
                         "meta": json.dumps({"en_link": en_post["link"], "nl_link": link, "en_post": en_post["id"],
                                             "nl_post": post["id"], "en_paragraph": i, "nl_paragraph": j,
                                             "align_score": round(score, 3), "author": en_page["author"],
                                             "nl_date": post["date"]}, ensure_ascii=False)})
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    Path(args.out + ".stats.json").write_text(json.dumps(dict(stats), indent=1) + "\n")
    print("GV_ALIGN " + json.dumps(dict(stats)), flush=True)


# ---------------------------------------------------------------------------------------------------- Europarl

# The speaker's role before the speech in an intervention's first paragraph ("lid van de Commissie . – ", "namens de
# PSE-Fractie . – "): metadata, in the Dutch files of 2004-2011 in 5,940 first paragraphs, in the English in 96
ROLE_PREFIX = re.compile(r"^(?P<role>[^.!?]{1,250}?)\s?\.\s?[–−—]\s+(?=\S)")
ROLE_WORDS_MAX = 30
# A role starts in lower case (Dutch: "namens", "lid van", "auteur"), or the speech after it opens with an address; a
# first sentence that ends in ". –" is no role ("Ik zeg dit: ten eerste. – Ten tweede, ...")
ADDRESS = re.compile(r"(?:Mijnheer|Mevrouw|Voorzitter|Mr|Mrs|Madam|President)\b")


def strip_role(text):
    m = ROLE_PREFIX.match(text)
    if not m or len(m.group("role").split()) > ROLE_WORDS_MAX:
        return text
    if m.group("role")[:1].islower() or ADDRESS.match(text[m.end():]):
        return text[m.end():]
    return text


def europarl(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, stats = [], collections.Counter()
    shards = sorted(Path(args.motra).glob("shard-*.parquet"))
    if not shards:
        raise SystemExit(f"no shard-*.parquet in {args.motra} (data.py motra)")
    for path in shards:
        for r in pq.read_table(path).to_pylist():
            stats["paired"] += 1
            if r["year"] < FROM_YEAR:
                stats["before_from_year"] += 1
                continue
            en, nl = r["en"], r["nl"]
            if r["paragraph"] == 0:
                en2, nl2 = strip_role(en), strip_role(nl)
                stats["role_stripped_en"] += en2 != en
                stats["role_stripped_nl"] += nl2 != nl
                en, nl = en2, nl2
            rows.append({"id": f"europarl:{r['id']}", "source": "europarl", "en": en, "nl": nl, "date": r["date"],
                         "year": r["year"],
                         "meta": json.dumps({"speaker": r["speaker"], "speaker_id": r["speaker_id"], "mode": r["mode"],
                                             "role": r["role"], "paragraph": r["paragraph"]}, ensure_ascii=False)})
    stats["kept"] = len(rows)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    Path(args.out + ".stats.json").write_text(json.dumps(dict(stats), indent=1) + "\n")
    print("EUROPARL " + json.dumps(dict(stats)), flush=True)


# ---------------------------------------------------------------------------------------------------- candidates

def real_fails(en, nl):
    """The hard checks a real pair fails (empty: it goes on to the query of sft_build.py)"""
    import publish

    fails = []
    if len(en.split()) < MIN_WORDS:
        fails.append("short")
    if " ".join(nl.split()).lower() == " ".join(en.split()).lower():
        fails.append("same")
    elif publish.english_left(nl, en):
        fails.append("english_left")
    if max(len(en), len(nl)) > LEN_RATIO_MAX * max(1, min(len(en), len(nl))):
        fails.append("len_ratio")
    return fails


def held_out(args):
    """(post ids, links) of Global Voices articles of the blind assessment and the idiom set"""
    ids, links = set(), set()
    for path in (args.blind, args.idioms):
        for r in read_jsonl(path):
            for key in ("link", "url"):
                if (r.get("meta") or {}).get(key) or r.get(key):
                    links.add(norm_link((r.get("meta") or {}).get(key) or r.get(key)))
            for value in (r.get("id"), r.get("pid"), (r.get("meta") or {}).get("pid")):
                if isinstance(value, str) and value.startswith("gv:"):
                    ids.add(int(value.split(":")[1]))
    return ids, links


def candidates(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.gv).to_pylist() + pq.read_table(args.europarl).to_pylist()
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("real pairs with the same id")
    blind_ids, blind_links = held_out(args)
    stats = collections.Counter()
    for r in rows:
        r["en"], r["nl"] = r["en"].strip(), r["nl"].strip()
        fails = real_fails(r["en"], r["nl"])
        if r["source"] == "globalvoices":
            meta = json.loads(r["meta"])
            if meta["en_post"] in blind_ids or norm_link(meta["en_link"]) in blind_links:
                fails.insert(0, "blind_article")
            elif r["date"][:4] == POOL_YEAR:
                fails.insert(0, "blind_pool_year")
        r["fails"] = fails
        r["words_en"], r["words_nl"] = len(r["en"].split()), len(r["nl"].split())
        r["chars_en"], r["chars_nl"] = len(r["en"]), len(r["nl"])
        stats[f"{r['source']}/pairs"] += 1
        for f in fails:
            stats[f"{r['source']}/alone/{f}"] += 1
        if fails:
            stats[f"{r['source']}/first/{fails[0]}"] += 1
        else:
            stats[f"{r['source']}/passed"] += 1
    pq.write_table(pa.Table.from_pylist(rows), out / "real.parquet")
    passed = [r for r in rows if not r["fails"]]
    for w in range(args.workers):
        part = passed[w::args.workers]
        pq.write_table(pa.table({"id": [r["id"] for r in part], "en": [r["en"] for r in part],
                                 "nl": [r["nl"] for r in part]}), out / f"metricx-in-w{w}.parquet")
    report = {"stats": dict(sorted(stats.items())), "blind_posts": sorted(blind_ids), "blind_links": sorted(blind_links),
              "min_words": MIN_WORDS, "len_ratio_max": LEN_RATIO_MAX, "from_year": FROM_YEAR, "pool_year": POOL_YEAR,
              "to_score": len(passed), "workers": args.workers}
    (out / "candidates.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("CANDIDATES " + json.dumps(report["stats"]), flush=True)


def join_qe(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(args.real).to_pylist()
    qe = {}
    for path in args.qe:
        t = pq.read_table(path).to_pydict()
        for i, v in zip(t["id"], t["qe_en_nl"]):
            if i in qe:
                raise SystemExit(f"{path}: {i} scored twice")
            qe[i] = v
    missing = 0
    for r in rows:
        r["qe_en_nl"] = qe.pop(r["id"], None)
        if not r["fails"] and r["qe_en_nl"] is None:
            missing += 1
    if qe:
        raise SystemExit(f"{len(qe)} scores of ids not in {args.real}")
    if missing:
        raise SystemExit(f"{missing} pairs that pass the hard checks have no MetricX-QE")
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"JOINED {sum(r['qe_en_nl'] is not None for r in rows)} QE scores -> {args.out}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="The real translation pairs of arm R2")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gv-fetch")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.5, help="seconds after each request, per thread")
    p = sub.add_parser("gv-align")
    p.add_argument("--fetched", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("europarl")
    p.add_argument("--motra", required=True, help="the shards of en-nl-judge's data.py motra")
    p.add_argument("--out", required=True)
    p = sub.add_parser("candidates")
    p.add_argument("--gv", required=True)
    p.add_argument("--europarl", required=True)
    p.add_argument("--blind", required=True, help="blind100.jsonl")
    p.add_argument("--idioms", required=True, help="idiom30.jsonl")
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--out", required=True)
    p = sub.add_parser("join-qe")
    p.add_argument("--real", required=True)
    p.add_argument("--qe", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"gv-fetch": gv_fetch, "gv-align": gv_align, "europarl": europarl, "candidates": candidates,
     "join-qe": join_qe}[args.command](args)


if __name__ == "__main__":
    main()
