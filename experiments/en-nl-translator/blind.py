#!/usr/bin/env python3
"""The 100 English passages of the blind assessment (protocol.md §4.2.4), fixed before the fine-tune: the 35 FLAN
evaluation rows of set `yeb`, 35 WMT24++ segments and 30 passages with an English idiom from Global Voices.

    python blind.py idioms --kaikki kaikki.org-dictionary-English.jsonl --out idioms.jsonl
    python blind.py gv-fetch --after 2025-01-01T00:00:00 --before 2026-01-01T00:00:00 --out posts.jsonl
    python blind.py candidates --posts posts.jsonl --idioms idioms.jsonl --out candidates.jsonl   (spaCy)
    python blind.py select --candidates candidates.jsonl --flan-eval eval-rows.parquet --evalsets evalsets.parquet \\
        --tagtest-items items.parquet --train SET [SET ...] --out DIR [--exclude PID ...]

idioms: the multiword English lemmas of kaikki.org's extract of the English Wiktionary that Wiktionary marks as an idiom
(a sense tagged idiomatic, or the category "English idioms"); per idiom whether Wiktionary also gives it a literal
sense (a sense tagged literally, or the gloss "Used other than figuratively or idiomatically: ...") or any sense not
marked idiomatic or figurative ("free lunch": a meal that costs nothing). select leaves both out: there a passage may
use the words literally.
gv-fetch: the English posts of Global Voices (globalvoices.org, CC BY 3.0) in a date range, from its WordPress API.
candidates: per post its own paragraphs (not quotes, captions or embeds; not a post that says it was published
elsewhere first or republished under a partnership: its licence may differ) of PASSAGE_WORDS words, and the idioms in
each, matched on lemma (spaCy en_core_web_sm): each word of the idiom by its form or lemma, a placeholder of the
idiom (one's, oneself, someone, something) by a possessive, a reflexive or a short gap.
select: rule_ok (below) for the idioms that may count; the idioms in an order fixed by IDIOM_SEED (each idiom once, so
that common phrases do not crowd the choice), and per idiom, in an order fixed by IDIOM_SEED/<idiom>, the first
passage of a post not chosen yet that shares no 13-gram with the training sets (--train) and whose page names its
author and the CC BY 3.0 licence, until N_IDIOM; N_WMT WMT24++ segments without is_bad_source and outside the tag
test's 300 (their translations are shown in examples-qe-rule.md), drawn with WMT_SEED; FLAN's 35 rows of set yeb.
--exclude leaves passages out by pid (a literal use found by reading), --exclude-idioms idioms ("a|b") that reading found
used literally passage after passage, so that the choice stays one command.
"""

import argparse
import collections
import hashlib
import html as html_lib
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_IDIOM, N_WMT, N_FLAN = 30, 35, 35
IDIOM_SEED = "en-nl-sft/blind/idiom"
WMT_SEED = "en-nl-sft/blind/wmt"
PASSAGE_WORDS = (40, 200)
GV_API = "https://globalvoices.org/wp-json/wp/v2/posts"
USER_AGENT = "mbert-jax en-nl-sft blind.py (research; contact: github.com/yhavinga)"
LICENCE = "creativecommons.org/licenses/by/3.0"

FUNCTION = frozenset("""a an the of to in on at by for with from up down out off over into onto as and or but if so
    not no nor be is are was were been being am have has had do does did it its this that these those there here what
    which who whom how when where why all any some one ones own than then too very just more most much many such
    i me you he him she her we us they them my your his our their 's s well also even still only ever never yet again
    quite rather""".split())
POSS_WORDS = frozenset("my your his her its our their one's whose".split())
REFL_WORDS = frozenset("myself yourself himself herself itself ourselves yourselves themselves oneself".split())
GAP_WORDS = frozenset("someone somebody something somewhere sb sth".split())
POSS_HEADS = frozenset("one someone somebody".split())
LITERAL_GLOSS = re.compile(r"^\s*used other than (?:figuratively|with a figurative)", re.I)
REPUBLISHED = re.compile(r"originally (?:published|appeared)|republished|content[- ]sharing|partnership agreement|"
                         r"under a (?:media )?partnership|is (?:re)?published here", re.I)
BOILERPLATE = re.compile(r"^\W*(?:read (?:more|also)|photo|image|screenshot|featured image|note:|source:|this (?:post|"
                         r"article|story)|editor['’]s note|follow |subscribe|translated|translation|watch|listen|see "
                         r"also|related)", re.I)
GAP_MAX = 4
HYPHENS = frozenset({"-", "–", "‐"})


# ---------------------------------------------------------------- the idiom list

def categories_of(obj):
    out = set()
    for c in obj.get("categories", []) or []:
        out.add(c.get("name") if isinstance(c, dict) else c)
    return out


def build_idioms(lines):
    """{word: {idiom, pos, literal, gloss}} of kaikki's English entries, merged over parts of speech"""
    found = {}
    for line in lines:
        if "idiom" not in line:  # an idiom has the tag idiomatic or the category English idioms
            continue
        e = json.loads(line)
        word = e.get("word", "")
        if e.get("lang_code") != "en" or len(word.split()) < 2:
            continue
        senses = [s for s in e.get("senses", []) if not s.get("form_of") and not s.get("alt_of")]
        if not senses:
            continue
        tags = [set(s.get("tags", [])) for s in senses]
        if all(t & {"archaic", "obsolete"} for t in tags):
            continue
        idiom_senses = [s for s, t in zip(senses, tags)
                        if "idiomatic" in t or "English idioms" in categories_of(s)]
        if not idiom_senses and "English idioms" not in categories_of(e):
            continue
        literal = any("literally" in t for t in tags) or any(
            LITERAL_GLOSS.match(g) for s in senses for g in s.get("glosses", []) or [])
        plain = any(not (t & {"idiomatic", "figuratively"} or "English idioms" in categories_of(s))
                    for s, t in zip(senses, tags))
        gloss = next((g for s in idiom_senses or senses for g in s.get("glosses", []) or []
                      if not LITERAL_GLOSS.match(g)), "")
        got = found.setdefault(word, {"idiom": word, "pos": [], "literal": False, "plain_sense": False, "gloss": gloss})
        got["pos"] = sorted(set(got["pos"]) | {e.get("pos")})
        got["literal"] = got["literal"] or literal
        got["plain_sense"] = got["plain_sense"] or plain
        got["gloss"] = got["gloss"] or gloss
    return found


def idioms(args):
    with open(args.kaikki, encoding="utf-8") as f:
        found = build_idioms(f)
    with open(args.out, "w", encoding="utf-8") as f:
        for word in sorted(found):
            f.write(json.dumps(found[word], ensure_ascii=False) + "\n")
    print(json.dumps({"idioms": len(found), "literal": sum(v["literal"] for v in found.values()),
                      "plain_sense": sum(v["plain_sense"] for v in found.values()),
                      "idiomatic_only": sum(not v["literal"] and not v["plain_sense"] for v in found.values())}))


# ---------------------------------------------------------------- Global Voices

def http_get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8"), dict(r.headers)
        except Exception as exc:  # noqa: BLE001 (retried, then raised with the URL)
            if k == tries - 1:
                raise SystemExit(f"GET {url}: {exc}") from exc
            time.sleep(5 * (k + 1))


def gv_fetch(args):
    fields = "id,date,link,title,content,author,categories,tags,modified"
    page, total, n = 1, None, 0
    with open(args.out, "w", encoding="utf-8") as f:
        while total is None or page <= total:
            url = (f"{GV_API}?per_page=100&page={page}&after={args.after}&before={args.before}&orderby=date&order=asc"
                   f"&_fields={fields}")
            body, headers = http_get(url)
            total = int(headers.get("X-WP-TotalPages") or headers.get("x-wp-totalpages"))
            for post in json.loads(body):
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
                n += 1
            page += 1
            time.sleep(1)
    print(json.dumps({"posts": n, "pages": total, "after": args.after, "before": args.before}))


def paragraphs(html, links=False):
    """The post's own paragraphs: <p> outside quotes, captions, embeds, tables and lists; not mostly italic (notes).
    With links, (text, the hrefs of its <a> elements) per paragraph (real_pairs.py aligns on them)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out = []
    for p in soup.find_all("p"):
        if p.find_parent(["blockquote", "figure", "figcaption", "table", "li"]):
            continue
        if p.find_parent(class_=re.compile(r"caption|embed|tweet|instagram|wp-block-quote")):
            continue
        if any("caption" in c for c in p.get("class") or []):
            continue
        for br in p.find_all("br"):
            br.replace_with("\n")
        text = " ".join(p.get_text().split())
        italic = sum(len(" ".join(i.get_text().split())) for i in p.find_all(["em", "i"], recursive=False))
        if not text or italic >= 0.8 * len(text):
            continue
        out.append((text, [a.get("href") for a in p.find_all("a") if a.get("href")]) if links else text)
    return out


def english_enough(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.95


# ---------------------------------------------------------------- matching on lemma

def norm(s):
    return s.lower().replace("’", "'")


def pattern_of(doc):
    """Slots of an idiom: ("lit", keys), ("poss",), ("refl",) or ("gap",); leading and trailing gaps (arguments)
    dropped. None when no word is left."""
    toks = [t for t in doc if not t.is_punct or t.text in ("'s", "’s")]  # hyphens too: well-to-do = well to do
    slots, k = [], 0
    while k < len(toks):
        t, low = toks[k], norm(toks[k].text)
        nxt = norm(toks[k + 1].text) if k + 1 < len(toks) else None
        if low in POSS_HEADS and nxt == "'s":
            slots.append(("poss",))
            k += 2
            continue
        if low == "one's":
            slots.append(("poss",))
        elif low == "oneself":
            slots.append(("refl",))
        elif low in GAP_WORDS:
            slots.append(("gap",))
        else:
            slots.append(("lit", frozenset({low, norm(t.lemma_)})))
        k += 1
    while slots and slots[0] == ("gap",):
        slots.pop(0)
    while slots and slots[-1] == ("gap",):
        slots.pop()
    return slots if any(s[0] == "lit" for s in slots) else None


def counts(slots):
    words = [min(s[1]) for s in slots if s[0] == "lit"]
    content = [w for w in words if w not in FUNCTION and len(w) >= 3 and re.search(r"[a-z]", w)]
    return len(slots), len(content)


def rule_ok(slots):
    """RULE: an idiom counts when it has at least two content words, or three or more slots with at least one
    content word (this keeps out grammatical phrases such as 'of course' and 'as well as')"""
    n, content = counts(slots)
    return content >= 2 or (n >= 3 and content >= 1)


class Matcher:
    def __init__(self, patterns):
        self.patterns = patterns  # [(idiom, slots)]
        self.index = collections.defaultdict(list)
        for k, (_, slots) in enumerate(patterns):
            first = next(i for i, s in enumerate(slots) if s[0] == "lit")
            for key in slots[first][1]:
                self.index[key].append((k, first))

    @staticmethod
    def _match(slots, toks, j):
        """end index of slots matched from toks[j], or None (first match, depth first)"""
        if not slots:
            return j
        kind = slots[0][0]
        if kind == "lit":
            if j < len(toks) and toks[j][1] & slots[0][1]:
                return Matcher._match(slots[1:], toks, j + 1)
            return None
        if kind == "refl":
            if j < len(toks) and toks[j][0] in REFL_WORDS:
                return Matcher._match(slots[1:], toks, j + 1)
            return None
        if kind == "poss":
            if j < len(toks) and toks[j][0] in POSS_WORDS:
                end = Matcher._match(slots[1:], toks, j + 1)
                if end is not None:
                    return end
            for span in range(1, GAP_MAX):  # a noun phrase of 1-3 words and 's
                k = j + span
                if k < len(toks) and toks[k][0] in ("'s", "'") and all(toks[i][2] for i in range(j, k)):
                    end = Matcher._match(slots[1:], toks, k + 1)
                    if end is not None:
                        return end
            return None
        for span in range(1, GAP_MAX + 1):  # gap
            if j + span > len(toks) or not all(toks[i][2] for i in range(j, j + span)):
                break
            end = Matcher._match(slots[1:], toks, j + span)
            if end is not None:
                return end
        return None

    def find(self, toks):
        """[(idiom, start, end)] in toks: (lower form, {form, lemma}, is a word) per token"""
        found = set()
        for i, (_, keys, _) in enumerate(toks):
            for key in keys:
                for k, first in self.index.get(key, ()):
                    idiom, slots = self.patterns[k]
                    starts = [i] if first == 0 else range(max(0, i - GAP_MAX * first), i + 1)
                    for s in starts:
                        end = self._match(slots, toks, s)
                        if end is not None and s <= i < end:
                            found.add((idiom, s, end))
                            break
        return sorted(found, key=lambda x: (x[1], x[0]))


def candidates(args):
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    entries = [json.loads(line) for line in open(args.idioms, encoding="utf-8")]
    patterns, stats = [], collections.Counter()
    meta = {}
    for e, doc in zip(entries, nlp.pipe([e["idiom"].replace("’", "'") for e in entries], batch_size=512)):
        slots = pattern_of(doc)
        if slots is None:
            stats["no_word"] += 1
            continue
        meta[e["idiom"]] = {"literal": e["literal"] or e["plain_sense"], "rule": rule_ok(slots), "gloss": e["gloss"],
                            "pos": e["pos"]}
        patterns.append((e["idiom"], slots))
    stats["patterns"] = len(patterns)
    stats["patterns_rule"] = sum(m["rule"] for m in meta.values())
    stats["patterns_rule_not_literal"] = sum(m["rule"] and not m["literal"] for m in meta.values())
    matcher = Matcher(patterns)
    posts = [json.loads(line) for line in open(args.posts, encoding="utf-8")]
    passages = []
    for post in posts:
        html = post["content"]["rendered"]
        if REPUBLISHED.search(re.sub(r"<[^>]+>", " ", html)):
            stats["posts_republished"] += 1
            continue
        stats["posts_used"] += 1
        for k, text in enumerate(paragraphs(html)):
            words = len(text.split())
            stats["paragraphs"] += 1
            if not PASSAGE_WORDS[0] <= words <= PASSAGE_WORDS[1]:
                continue
            if BOILERPLATE.match(text) or not english_enough(text) or not re.search(r"[.!?…\"”’)]$", text):
                stats["paragraphs_boilerplate"] += 1
                continue
            passages.append({"pid": f"gv:{post['id']}:{k}", "post_id": post["id"], "link": post["link"],
                             "date": post["date"], "title": post["title"]["rendered"], "paragraph": k,
                             "words": words, "text": text})
    stats["passages"] = len(passages)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, doc in zip(passages, nlp.pipe([p["text"] for p in passages], batch_size=256)):
            # hyphens out, as in the idioms' patterns
            kept = [t for t in doc if t.text not in HYPHENS]
            toks = [(norm(t.text), frozenset({norm(t.text), norm(t.lemma_)}), not t.is_punct and not t.is_space)
                    for t in kept]
            hits = []
            for idiom, s, e in matcher.find(toks):
                span = doc[kept[s].i:kept[e - 1].i + 1]
                hits.append({"idiom": idiom, "span": span.text, "start": span.start_char, "end": span.end_char,
                             **meta[idiom]})
            if hits:
                stats["passages_with_idiom"] += 1
                if any(h["rule"] and not h["literal"] for h in hits):
                    stats["passages_with_countable_idiom"] += 1
            f.write(json.dumps(p | {"idioms": hits}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats)))
    Path(args.out + ".stats.json").write_text(json.dumps(dict(stats), indent=1) + "\n")


# ---------------------------------------------------------------- the choice

def page_facts(link):
    """(author, licence link found) of a post's page: its meta author and the CC BY 3.0 link of its footer"""
    html, _ = http_get(link)
    m = re.search(r'<meta name="author" content="([^"]+)"', html)
    lic = re.search(r"Creative Commons Attribution 3\.0", html) is not None and LICENCE in html
    return (html_lib.unescape(m.group(1)).strip() if m else None), lic


def train_overlap(texts_by_id, train_dirs):
    """{id: number of training examples sharing a 13-gram with it}, over the English and Dutch of every training
    example of the sets (prompt text, context turns, Dutch target)"""
    import pyarrow.parquet as pq
    import select_passages as sp

    owner = collections.defaultdict(set)
    for i, texts in texts_by_id.items():
        for t in texts:
            for g in sp.ngrams(t):
                owner[g].add(i)
    hits = collections.Counter()
    scanned = 0
    for d in train_dirs:
        meta = pq.read_table(Path(d) / "train-meta.parquet", columns=["en", "context"]).to_pylist()
        msgs = pq.read_table(Path(d) / "train.parquet").column("messages").to_pylist()
        for m, row in zip(meta, msgs):
            parts = [m["en"], row[1]["content"]] + [c[k] for c in m["context"] or [] for k in ("source", "translation")]
            seen = set()
            for part in parts:
                for g in sp.ngrams(part or ""):
                    seen |= owner.get(g, set())
            for i in seen:
                hits[i] += 1
            scanned += 1
    return hits, scanned


def select(args):
    import pyarrow.parquet as pq

    cands = [json.loads(line) for line in open(args.candidates, encoding="utf-8")]
    excluded = set(args.exclude or [])
    excluded_idioms = {i.strip() for i in (args.exclude_idioms or "").split("|") if i.strip()}
    known = {h["idiom"] for p in cands for h in p["idioms"]}
    if excluded_idioms - known:
        raise SystemExit(f"--exclude-idioms: not in the candidates: {sorted(excluded_idioms - known)}")
    for p in cands:
        p["countable"] = sorted({h["idiom"] for h in p["idioms"] if h["rule"] and not h["literal"]} - excluded_idioms)
    pool = sorted((p for p in cands if p["countable"]), key=lambda p: p["pid"])
    # WMT24++: outside the tag test's segments (shown with translations in examples-qe-rule.md)
    tagtest = {r["id"] for r in pq.read_table(args.tagtest_items, columns=["id"]).to_pylist()}
    wmt_all = [r for r in pq.read_table(args.evalsets).to_pylist()
               if r["set"] == "wmt24pp" and not json.loads(r["meta"]).get("is_bad_source")]
    wmt_pool = sorted((r for r in wmt_all if r["id"] not in tagtest), key=lambda r: r["id"])
    wmt = sorted(random.Random(WMT_SEED).sample(wmt_pool, N_WMT), key=lambda r: r["id"])
    flan = sorted((r for r in pq.read_table(args.flan_eval).to_pylist() if r["set"] == "yeb"),
                  key=lambda r: (r["host"], r["position"]))
    if len(flan) != N_FLAN:
        raise SystemExit(f"{len(flan)} FLAN rows of set yeb, not {N_FLAN}")
    # 13-grams with the training sets: every candidate of the pool, the WMT24++ segments (source and reference), FLAN
    texts = {p["pid"]: [p["text"]] for p in pool}
    texts |= {r["id"]: [r["source"], r["reference"]] for r in wmt}
    texts |= {f"flan:{r['host']}-{r['position']}": [r["inputs"]] for r in flan}
    overlap, scanned = train_overlap(texts, args.train)
    log = collections.Counter()
    by_idiom = collections.defaultdict(list)
    for p in pool:
        for i in p["countable"]:
            by_idiom[i].append(p)
    order = sorted(by_idiom)
    random.Random(IDIOM_SEED).shuffle(order)
    chosen, posts = [], set()
    for idiom in order:
        if len(chosen) == N_IDIOM:
            break
        if any(idiom in {h["idiom"] for h in c["expressions"]} for c in chosen):
            log["idiom_in_a_chosen_passage"] += 1
            continue
        options = sorted(by_idiom[idiom], key=lambda p: p["pid"])
        random.Random(f"{IDIOM_SEED}/{idiom}").shuffle(options)
        for p in options:
            if p["post_id"] in posts:
                log["same_post"] += 1
                continue
            if p["pid"] in excluded:
                log["excluded"] += 1
                continue
            if overlap[p["pid"]]:
                log["train_13gram"] += 1
                continue
            author, licensed = page_facts(p["link"])
            time.sleep(1)
            if not author or not licensed:
                log["page_without_author_or_licence"] += 1
                continue
            hits = [h for h in sorted(p["idioms"], key=lambda h: h["idiom"] != idiom) if h["idiom"] in p["countable"]]
            hits = [h for k, h in enumerate(hits) if h["idiom"] not in {x["idiom"] for x in hits[:k]}]  # once each
            chosen.append(p | {"author": author, "licence": "CC BY 3.0", "expressions": hits, "drawn": idiom})
            posts.add(p["post_id"])
            break
        else:
            log["idiom_without_a_usable_passage"] += 1
    if len(chosen) != N_IDIOM:
        raise SystemExit(f"only {len(chosen)} idiom passages: {dict(log)}")
    wmt_overlap = {r["id"]: overlap[r["id"]] for r in wmt if overlap[r["id"]]}
    if wmt_overlap:
        raise SystemExit(f"WMT24++ segments share a 13-gram with the training sets: {wmt_overlap}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idiom_rows = [{"id": p["pid"], "set": "idiom", "source": p["text"],
                   "meta": {"url": p["link"], "author": p["author"], "licence": "CC BY 3.0",
                            "title": p["title"], "date": p["date"], "paragraph": p["paragraph"], "words": p["words"],
                            "drawn": p["drawn"],
                            "expressions": [{"idiom": h["idiom"], "span": h["span"], "gloss": h["gloss"]}
                                            for h in p["expressions"]]}} for p in chosen]
    rows = [{"id": f"flan:{r['host']}-{r['position']}", "set": "flan", "source": r["inputs"].strip(),
             "meta": {"task": r["source"]}} for r in flan]
    rows += [{"id": r["id"], "set": "wmt24pp", "source": r["source"].strip(),
              "meta": {"domain": json.loads(r["meta"]).get("domain"), "reference": r["reference"]}} for r in wmt]
    rows += sorted(idiom_rows, key=lambda r: r["id"])
    for k, r in enumerate(rows, 1):
        r["blind_id"] = f"blind-{k:03d}"
    write_jsonl(out / "idiom30.jsonl", [r for r in rows if r["set"] == "idiom"])
    write_jsonl(out / "blind100.jsonl", rows)
    report = {"pool": len(pool), "idioms_in_pool": len(by_idiom), "candidates": len(cands), "skipped": dict(log),
              "excluded": sorted(excluded), "excluded_idioms": sorted(excluded_idioms),
              "train_examples_scanned": scanned, "train": args.train,
              "train_13gram_overlap": {"idiom_chosen": sum(overlap[p["pid"]] > 0 for p in chosen),
                                       "wmt": len(wmt_overlap),
                                       "flan_rows_sharing": sum(overlap[r["id"]] > 0 for r in rows
                                                                if r["set"] == "flan")},
              "wmt_pool": len(wmt_pool), "wmt_all": len(wmt_all), "tagtest_excluded": len(tagtest & {r["id"] for r in wmt_all}),
              "seeds": {"idiom": IDIOM_SEED, "wmt": WMT_SEED},
              "sha256": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("idiom30.jsonl", "blind100.jsonl")}}
    (out / "select-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    stats = json.loads(Path(args.candidates + ".stats.json").read_text())
    manifest = Path(args.manifest).read_text(encoding="utf-8") if args.manifest else ""
    (out / "blind-idiom-30.md").write_text(blind_md(report, stats, manifest, rows, args), encoding="utf-8")
    print(json.dumps(report))


def blind_md(report, stats, manifest, rows, args):
    idiom_rows = [r for r in rows if r["set"] == "idiom"]
    n = report["skipped"]
    lines = [
        "# De 30 idioompassages van de blinde beoordeling (en de lijst van 100)", "",
        "Gekozen op 10 nov 2026, vóór de fine-tune, met `blind.py` (branch `en-nl-sft`) op mbert-mix; protocol §4.2.4. "
        "De beoordeling doet een verse Claude-subagent; twijfelgevallen gaan naar Yeb (besluit 10 nov ~09:20). "
        "**De passages zelf staan onderaan, ingeklapt: niet lezen vóór de beoordeling.**", "",
        "## Bron en licentie", "",
        "- Engelse artikelen van Global Voices (`globalvoices.org`), opgehaald via de WordPress-API van de site "
        f"(`{GV_API}`), alle berichten met een datum vanaf {args.after_label}: "
        f"{stats.get('posts_used', 0) + stats.get('posts_republished', 0)} berichten, waarvan "
        f"{stats.get('posts_republished', 0)} weggelaten omdat de tekst zegt dat het stuk elders eerst verscheen of "
        "onder een samenwerking is overgenomen (dan kan een andere licentie gelden).",
        "- Licentie: de voettekst van elke pagina van Global Voices noemt \"Creative Commons Attribution 3.0\" met een "
        f"link naar `https://{LICENCE}/` [gecontroleerd per gekozen artikel op de pagina zelf]. Per passage staan de URL "
        "en de auteur (de `meta name=\"author\"` van de pagina) hieronder; dat is de vermelding die CC BY 3.0 vraagt.",
        "- Niet de OPUS-versie van Global Voices: daar staan geen URL en auteur per artikel bij, en de nieuwste versie "
        "loopt tot 2018.", "",
        "## Idioomlijst en matching", "",
        "- Lijst: de Engelse Wiktionary, via het extract van kaikki.org (dezelfde bron als de Nederlandse idioomlijst "
        "van de selectie): Engelse lemma's van twee of meer woorden met een betekenis die Wiktionary `idiomatic` noemt "
        "of in de categorie \"English idioms\" zet.",
        f"- Weggelaten (letterlijk gebruik waar mogelijk uitsluiten): elke uitdrukking waarvoor Wiktionary ook een "
        "letterlijke betekenis geeft (tag `literally` of de glosse \"Used other than figuratively or idiomatically\"), "
        "of een betekenis die niet als idioom of figuurlijk is gemarkeerd (\"free lunch\": een gratis maaltijd); "
        "en grammaticale vaste verbindingen (regel: minstens twee inhoudswoorden, of drie of meer delen met minstens één "
        f"inhoudswoord; zo vallen \"of course\" en \"as well as\" af). Patronen: {stats.get('patterns')}, waarvan "
        f"{stats.get('patterns_rule')} volgens de regel, {stats.get('patterns_rule_not_literal')} daarvan zonder "
        "letterlijke betekenis.",
        "- Matching op lemma met spaCy `en_core_web_sm`: elk woord van de uitdrukking op vorm of lemma; one's/someone's "
        "op een bezittelijk voornaamwoord of een naam met 's; oneself op een wederkerend voornaamwoord; "
        "someone/something op 1–4 woorden; zulke argumenten aan begin of eind vallen weg.",
        f"- Passages: één eigen alinea van {PASSAGE_WORDS[0]}–{PASSAGE_WORDS[1]} woorden (geen citaat, onderschrift, "
        "embed, lijst of notitie in cursief; Engels; eindigt op een leesteken). Alinea's: "
        f"{stats.get('paragraphs')}; passages: {stats.get('passages')}; met een uitdrukking: "
        f"{stats.get('passages_with_idiom')}; met een uitdrukking die telt: {stats.get('passages_with_countable_idiom')}.",
        f"- Keuze: eerst de uitdrukking, dan de passage, zodat alledaagse verbindingen (\"in the world\") de keuze niet "
        f"vullen. De {report['idioms_in_pool']} uitdrukkingen die tellen en in een passage voorkomen, in een volgorde "
        f"vastgelegd met seed `{IDIOM_SEED}`; per uitdrukking de passages in een volgorde met seed "
        f"`{IDIOM_SEED}/<uitdrukking>`, en daarvan de eerste uit een artikel dat nog niet gekozen is, die geen 13-gram "
        "deelt met de trainingsdata en waarvan de pagina auteur en licentie noemt; tot 30. Overgeslagen: "
        + ", ".join(f"{k} {v}" for k, v in n.items()) + ".",
        f"- Met de hand uitgesloten na het nalezen van de gekozen passages (letterlijk gebruik of een andere betekenis "
        f"dan het idioom; gelezen door de subagent die koos): passages {', '.join(report['excluded']) or 'geen'}; "
        f"uitdrukkingen die telkens letterlijk bleken: "
        f"{', '.join(chr(34) + i + chr(34) for i in report['excluded_idioms']) or 'geen'}.", "",
        "## Controle op de trainingsdata (13-gram)", "",
        f"Tegen alle {report['train_examples_scanned']} trainvoorbeelden van {', '.join(report['train'])} (Engels, "
        "Nederlands en contextbeurten): gekozen idioompassages met een gedeeld 13-gram: "
        f"{report['train_13gram_overlap']['idiom_chosen']}; WMT24++-segmenten: {report['train_13gram_overlap']['wmt']}; "
        f"FLAN-rijen (ter informatie; zij liggen vast en blijven via hun segmenten buiten de replay): "
        f"{report['train_13gram_overlap']['flan_rows_sharing']} van 35.", "",
        "## De lijst van 100", "",
        f"`gs://t5-dutch-english-usc2/mbert/translator/sft/blind/blind100.jsonl` (sha256 "
        f"`{report['sha256']['blind100.jsonl']}`): 35 FLAN-rijen (set `yeb`), 35 WMT24++-segmenten (seed `{WMT_SEED}`, "
        f"uit de {report['wmt_all']} zonder `is_bad_source` min de {report['tagtest_excluded']} van de tagproef, dus uit "
        f"{report['wmt_pool']}), 30 idioompassages (`idiom30.jsonl`, sha256 `{report['sha256']['idiom30.jsonl']}`).", "",
        "Bronnen van de gegevens:", "", "```", manifest.strip(), "```", "",
        "<details><summary>De 30 passages (niet lezen vóór de beoordeling)</summary>", ""]
    for r in idiom_rows:
        m = r["meta"]
        expr = "; ".join(f"\"{e['idiom']}\" (in de tekst: \"{e['span']}\"; Wiktionary: {e['gloss']})"
                         for e in m["expressions"])
        lines += [f"### {r['blind_id']}: {r['id']}", "", f"- URL: {m['url']}", f"- Auteur: {m['author']} (CC BY 3.0)",
                  f"- Uitdrukking: {expr}", "", "> " + r["source"], ""]
    lines += ["</details>", ""]
    return "\n".join(lines)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="The passages of the blind assessment")
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("idioms")
    i.add_argument("--kaikki", required=True)
    i.add_argument("--out", required=True)
    g = sub.add_parser("gv-fetch")
    g.add_argument("--after", required=True)
    g.add_argument("--before", required=True)
    g.add_argument("--out", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--posts", required=True)
    c.add_argument("--idioms", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--candidates", required=True)
    s.add_argument("--flan-eval", required=True)
    s.add_argument("--evalsets", required=True)
    s.add_argument("--tagtest-items", required=True)
    s.add_argument("--train", nargs="+", required=True, help="set directories with train.parquet, train-meta.parquet")
    s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--exclude-idioms", default="", help="idioms to leave out, separated by |")
    s.add_argument("--manifest", default=None, help="the sources' manifest of run_blind.sh prepare")
    s.add_argument("--after-label", default="2025-01-01 tot 2026-01-01")
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"idioms": idioms, "gv-fetch": gv_fetch, "candidates": candidates, "select": select}[args.command](args)


if __name__ == "__main__":
    main()
