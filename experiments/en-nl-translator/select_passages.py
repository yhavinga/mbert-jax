#!/usr/bin/env python3
"""Step 0 of the en-nl translator (hammond mbert/designs/en-nl-translator.md, §3): original Dutch passages to
back-translate, from the Handelingen of the Tweede Kamer and three collections of GPT-NL.

    python select_passages.py idioms --kaikki kaikki-dutch.jsonl --out idioms.json
    python select_passages.py select --out <dir> --n 2000 --n-eval 0 --idioms idioms.json \\
        --wordlist opentaal-wordlist.txt --tokenizer <TranslateGemma tokenizer> --processes 64

Sources (SOURCES), read straight from Hugging Face, only the row groups and columns needed:
- handelingen: yhavinga/dutch-handelingen-modern, rows with chamber == tweede_kamer; a passage stays inside one turn
  (`turns`) of a speaker (not null), never of the chair (a speaker name with "voorzitter") or of Vraag/Antwoord.
- raad, dpc, rechtspraak: tvosch/GPT-NL-propella-annotations, collections openraadsinformatie,
  dienst_publiek_en_communicatie and de_rechtspraak; documents of content_quality good/excellent, content_integrity
  complete/mostly_complete, and a content_type besides boilerplate and structured_data. Council documents that are
  conversational or opinion_editorial weigh RAAD_PREFERRED_WEIGHT times as much in the draw (the design's "bij
  voorkeur"; the council minutes themselves are mostly legal_document/procedural, so the others stay in).

A passage is 1-4 consecutive prose paragraphs of one block (a turn, or a document) with MIN_TOKENS-MAX_TOKENS Gemma
tokens; the number of paragraphs aimed at is drawn per passage, so lengths vary. A paragraph longer than MAX_TOKENS is
cut into runs of whole sentences, each a passage of its own (`piece` = true). In council documents a speaker label at
the start of a paragraph is removed and a passage never runs over the next label (no change of speaker inside one).

Filters per passage (all recorded, a passage passes when none fails): langid not nl, an old spelling form
(OLD_SPELLING), the share of lowercase words outside OpenTaal's word list, the share of digits and of punctuation among
the characters, the share of capitalized words (lists of names), motion boilerplate (MOTION), and in GPT-NL a name
the PII filter made inconsistent (name_conflict), and in council documents a speaker label inside a paragraph
(speaker_change).

Idiom score: matches of the idiom list (multiword Dutch lemmas of the English Wiktionary, by kaikki.org; the verb of a
verbal idiom in any of its conjugated forms, separable ones too: "hakte de knoop door") per 100 words; idioms found
in more than MAX_IDIOM_SHARE of the passages are fixed phrases and do not count. Per source half
of the passages come from the highest idiom class (the top IDIOM_CLASS_FACTOR x the number needed, score > 0), the
other half at random from all that pass.

The split is per document (a hash of source and document id): eval passages come from other documents than the
training passages. Exact duplicates and passages that share a 13-gram of words with one already chosen are skipped,
and at most --max-per-doc passages come from one document.

Output in --out: passages.parquet (training side), eval.parquet (--n-eval), segments-w<k>.parquet for translate.py
(id, text, tokens; every worker a similar mix of lengths), rejected.parquet (a sample per filter, for reading),
report.json (counts per step and per filter, the distributions, the row groups read, the resources' sha256).
"""

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HANDELINGEN = ("yhavinga/dutch-handelingen-modern", "abdba14ac9271b6d43ccefe6143202b7bdd4e900")
PROPELLA = ("tvosch/GPT-NL-propella-annotations", "db7f5947be4445e396aa0dd0df52ab6c1f418efc")
SOURCES = {
    "handelingen": {"repo": HANDELINGEN, "files": [f"data/train-{i:05d}-of-00008.parquet" for i in range(8)],
                    "columns": ["id", "chamber", "date", "title", "turns"]},
    "raad": {"repo": PROPELLA, "files": ["openraadsinformatie/train-00000-of-00001.parquet"]},
    "dpc": {"repo": PROPELLA, "files": ["dienst_publiek_en_communicatie/train-00000-of-00001.parquet"]},
    "rechtspraak": {"repo": PROPELLA, "files": ["de_rechtspraak/train-00000-of-00001.parquet"]},
}
PROPELLA_COLUMNS = ["doc_id", "text", "content_quality", "content_integrity", "content_type", "content_ratio"]
SHARES = {"handelingen": 0.5, "raad": 0.3, "dpc": 0.1, "rechtspraak": 0.1}
GOOD_QUALITY = {"good", "excellent"}
GOOD_INTEGRITY = {"complete", "mostly_complete"}
NOT_CONTENT = {"boilerplate", "structured_data"}
RAAD_PREFERRED = {"conversational", "opinion_editorial"}
RAAD_PREFERRED_WEIGHT = 3.0

MIN_TOKENS, MAX_TOKENS = 48, 384
MAX_PARAGRAPHS = 4
IDIOM_CLASS_FACTOR = 3
NGRAM = 13
# Stage-0 thresholds, chosen on the pilot's distributions (report.json has them per source)
MAX_OOV = 0.15
MAX_DIGITS = 0.05
MAX_PUNCT = 0.10
# Lists of names ("Aanwezig: dhr. Balke (D66), mw. ...") pass as prose; running text has far fewer capitalized words
MAX_CAPITALIZED = 0.30
# Idioms in more passages than this share are fixed phrases of the register ("sprake zijn van", "met name", "dank u
# wel"), not the figurative language the score is after; they are left out of the score (report.json lists them)
MAX_IDIOM_SHARE = 0.001

# The forms of §2 of the design and their kin: spelling before 1947, never in modern text
OLD_SPELLING = frozenset("""
    zoo zooals zoodat zoowel zooveel mensch menschen menschelijk eene hunne zijnen mijnen uwen onzen wensch wenschen
    wenschelijk oogenblik oogenblikken tusschen eenige eenigen duitschland visch vleesch gelooven hooren loopen
    koopen""".split())
# Motions read out in a debate or a council: formulaic, not speech
MOTION = re.compile(r"gehoord de beraadslaging|(?:^|\n)\s*(?:constaterende|overwegende|van mening|verzoekt de "
                    r"regering|draagt het college op)\b|gaat over tot de orde van de dag|deze motie is voorgesteld "
                    r"door|wordt de indiening ervan voldoende ondersteund", re.I)
SPEAKER_SKIP = re.compile(r"voorzitter|^\s*(?:vraag|antwoord)\b", re.I)

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
NGRAM_WORD = re.compile(r"\w+")
PUNCT = re.compile(r"[^\w\s]")
ABBREVIATIONS = frozenset("""mr dr drs ir ing prof art nr nrs blz jl mw mevr dhr ca bijv vgl o.a e.d i.c z.g.a etc
    st lid sub resp jr sr ds pag""".split())
SENTENCE_END = re.compile(r"(?<=[.!?…])(['\"”’)]*)\s+(?=[\"'“‘(]?[A-Z0-9À-Ý])")
PROSE_END = re.compile(r"[.!?…]['\"”’)]*$")
BULLET = re.compile(r"^(?:[-•*–·▪>#]|\(?(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,4})[.)]\s)")
URL = re.compile(r"https?://|www\.|\S+@\S+\.\w+")

# Speaker labels of council minutes: a title or a name with a party, then a colon
_PARTICLE = r"(?:[Vv]an|[Dd]e|[Dd]er|[Dd]en|[Tt]er|[Tt]en|[Hh]et|'t|[Oo]p|[Ii]n|[Ll]a|[Ll]e|[Dd]u|[Dd]a|[Vv]on|[Zz]u)"
_NAME = rf"(?:[A-Z][\w'’\-]*\.?|{_PARTICLE})"
_TITLE = (r"(?:[Dd]e\s+heer|[Dd]hr\.?|[Mm]evrouw|[Mm]evr\.?|[Mm]w\.?|[Mm]eneer|(?:[Dd]e\s+)?[Vv]oorzitter|"
          r"[Ww]ethouder|[Bb]urgemeester|[Gg]riffier|[Ii]nspreker|[Rr]aadslid|[Ss]ecretaris)")
_LABEL = (rf"(?:{_TITLE}(?:\s+{_NAME}){{0,5}}\s*(?:\([^)\n]{{1,30}}\))?|"
          rf"[A-Z][\w'’\-]+(?:\s+{_NAME}){{0,4}}\s*\([^)\n]{{1,30}}\))\s*:\s+")
SPEAKER_LABEL = re.compile(rf"^{_LABEL}")
# A label after a sentence inside a paragraph: another speaker within the passage
INLINE_LABEL = re.compile(rf"(?<=[.!?…])\s+{_LABEL}")
# Persons with a gendered title, for the names the PII filter of GPT-NL swapped ("Jet Stook-Bouwhuisen ... Hij")
_MALE = r"(?:[Dd]e\s+heer|[Dd]hr\.?|[Mm]eneer)"
_FEMALE = r"(?:[Mm]evrouw|[Mm]evr\.?|[Mm]w\.?)"
PERSON = re.compile(rf"(?<![\w-])({_MALE}|{_FEMALE})\s+((?:[A-Z]\.\s*)*(?:{_PARTICLE}\s+)*[A-Z][\w'’\-]+)")
FEMALE_TITLE = re.compile(rf"^{_FEMALE}$")
MALE_PRONOUN_START = re.compile(r"^(?:Hij|Hem)\b")


def words_of(text):
    return WORD.findall(text)


def split_sentences(text, abbreviations=ABBREVIATIONS):
    """Sentences of a paragraph: a cut after . ! ? before a capital or digit, not after a known abbreviation"""
    parts, start = [], 0
    for m in SENTENCE_END.finditer(text):
        before = text[start:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if last in abbreviations or (len(last) == 1 and last.isalpha()):
            continue
        parts.append(text[start:m.start() + len(m.group(1))].strip())
        start = m.end()
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def is_prose(paragraph):
    """A paragraph of running text: five words or more, ending a sentence, no bullet, table, header, URL or capitals"""
    if len(paragraph.split()) < 5 or not PROSE_END.search(paragraph) or BULLET.match(paragraph):
        return False
    if "|" in paragraph or URL.search(paragraph):
        return False
    letters = [c for c in paragraph if c.isalpha()]
    return sum(c.isupper() for c in letters) <= 0.25 * len(letters)


def clean(text):
    return re.sub(r"[ \t   ]+", " ", text).strip()


def unwrap(lines, wordset):
    """Lines of a paragraph that a PDF wrapped, joined; a hyphen at a line end goes when the word without it is in the
    word list (ge-/meente), and stays otherwise (sloop-/nieuwbouw)"""
    out = ""
    for line in (clean(x) for x in lines):
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            head = re.search(r"([^\W\d_]+)-$", out)
            tail = re.match(r"[^\W\d_]+", line)
            if head and tail and (head.group(1) + tail.group(0)).lower() in wordset:
                out = out[:-1] + line
                continue
            out += line
        else:
            out = f"{out} {line}" if out else line
    return out


def paragraphs_of(source, text, wordset):
    """The paragraphs of a document of a GPT-NL collection, cleaned"""
    if source == "raad":
        return [p for p in (unwrap(block.split("\n"), wordset) for block in re.split(r"\n\s*\n", text)) if p]
    if source == "rechtspraak":
        parts = []
        for block in re.split(r"\s{2,}|\n+", text):
            # numbered points (1.2.) and the headings of a ruling's parts (Titel 2) begin a paragraph
            parts += re.split(r"(?<=[.;:])\s+(?=(?:\d+(?:\.\d+)+\.?|Titel\s+\d+)\s)", block)
        return [p for p in (clean(re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", x.strip())) for x in parts) if p]
    return [p for p in (clean(x) for x in text.split("\n")) if p]


def name_conflict(text):
    """A name the PII filter swapped, where the passage shows it: one surname with both a male and a female title, or
    a sentence that opens with a woman by title (and names no man) followed by one that opens with Hij/Hem. (Zij and
    haar are left alone: plural and organizations use them too.)"""
    genders = collections.defaultdict(set)
    for title, name in PERSON.findall(text):
        genders[name.split()[-1]].add("f" if FEMALE_TITLE.match(title) else "m")
    if any(len(g) > 1 for g in genders.values()):
        return True
    sentences = split_sentences(text)
    for first, second in zip(sentences, sentences[1:]):
        m = PERSON.match(first)
        if (m and FEMALE_TITLE.match(m.group(1)) and MALE_PRONOUN_START.match(second)
                and all(FEMALE_TITLE.match(t) for t, _ in PERSON.findall(first))):
            return True
    return False


# ---------------------------------------------------------------- idioms

ARTICLES = frozenset({"de", "het", "een", "'t", "'n", "t", "n"})
PLACEHOLDERS = frozenset("""iets iemand iemands ergens zich zichzelf zijn z'n haar hun je jezelf jouw mijn men wat
    elkaar""".split())
# Words too common to make an idiom by themselves
FUNCTION = frozenset("""aan als bij dan dat die dit door en er in is met na naar niet nog of om ook op over te tot uit
    van voor zo zoals toe al af weg mee heen hier daar waar wel geen meer veel goed hem hen hij zij ze wij we ik jij
    jou mij ons""".split())
IDIOM_POS = {"verb", "phrase", "prep_phrase", "proverb"}
IDIOM_POS_TAGGED = {"noun", "adj", "adv", "intj"}  # only when Wiktionary tags a sense idiomatic


def idiom_tokens(text):
    return [w.lower().replace("’", "'") for w in WORD.findall(text)]


def build_idioms(kaikki_lines):
    """[{idiom, pos, keys, verb_forms}] from kaikki.org's Dutch entries: multiword lemmas of IDIOM_POS (and of
    IDIOM_POS_TAGGED when a sense is tagged idiomatic), not inflected forms, not archaic or obsolete in every sense.
    keys are the words that must occur (articles and placeholders left out); verb_forms are the conjugated forms of
    the verb of a verbal idiom (its last word), each a tuple of words ("hakte", "door")."""
    entries, verbs = [], {}
    for line in kaikki_lines:
        e = json.loads(line)
        if e.get("lang_code") != "nl":
            continue
        word, pos, senses = e.get("word", ""), e.get("pos"), e.get("senses", [])
        if any(s.get("form_of") or s.get("alt_of") for s in senses):
            continue
        if pos == "verb" and " " not in word:
            forms = {f["form"] for f in e.get("forms", []) if re.fullmatch(r"[a-zà-ÿ']+(?: [a-zà-ÿ']+)?", f["form"])
                     and not set(f.get("tags", [])) & {"table-tags", "inflection-template", "class"}}
            verbs.setdefault(word, set()).update(forms | {word})
        if " " not in word:
            continue
        tags = [set(s.get("tags", [])) for s in senses]
        if tags and all(t & {"archaic", "obsolete"} for t in tags):
            continue
        idiomatic = any("idiomatic" in t for t in tags)
        if pos in IDIOM_POS or (pos in IDIOM_POS_TAGGED and idiomatic):
            entries.append((word, pos))
    idioms, seen = [], set()
    for word, pos in entries:
        toks = idiom_tokens(word)
        if len(toks) < 2 or (word, pos) in seen:
            continue
        seen.add((word, pos))
        verb = toks[-1] if pos == "verb" else None
        rest = toks[:-1] if verb else toks
        keys = [t for t in rest if t not in ARTICLES and t not in PLACEHOLDERS]
        if not any(len(k) >= 3 and k not in FUNCTION for k in keys):
            continue
        forms = sorted(verbs.get(verb, {verb})) if verb else []
        idioms.append({"idiom": word, "pos": pos, "keys": keys, "verb_forms": [f.split() for f in forms],
                       "contiguous": None if verb else [None if t in PLACEHOLDERS else t for t in toks]})
    return idioms


class IdiomMatcher:
    """Idioms found in a text, sentence by sentence. A verbal idiom matches when its keys and one form of its verb
    (all words of a separable form) occur in one sentence within a window of WINDOW + len(keys) words, in any order
    (Dutch moves the verb); any other idiom matches as a run of words, a placeholder matching any word."""

    WINDOW = 12

    def __init__(self, idioms):
        self.idioms = idioms
        self.by_anchor = collections.defaultdict(list)
        for k, idiom in enumerate(idioms):
            content = [w for w in idiom["keys"] if len(w) >= 3 and w not in FUNCTION]
            self.by_anchor[max(content, key=len)].append(k)

    def _matches(self, idiom, toks, positions):
        if idiom["contiguous"] is not None:
            pattern, n = idiom["contiguous"], len(idiom["contiguous"])
            return any(all(p is None or p == toks[i + j] for j, p in enumerate(pattern))
                       for i in range(len(toks) - n + 1))
        keys = idiom["keys"]
        if not all(k in positions for k in keys):
            return False
        window = self.WINDOW + len(keys)
        for form in idiom["verb_forms"]:
            if not all(w in positions for w in form):
                continue
            spots = [positions[w] for w in keys + form]
            # the nearest occurrences: try each position of the first key as the centre
            for centre in spots[0]:
                near = [min(s, key=lambda p: abs(p - centre)) for s in spots]
                if max(near) - min(near) <= window:
                    return True
        return False

    def find(self, text):
        found = []
        for sentence in split_sentences(text):
            toks = idiom_tokens(sentence)
            positions = collections.defaultdict(list)
            for i, t in enumerate(toks):
                positions[t].append(i)
            hits = {k for t in positions for k in self.by_anchor.get(t, ())}
            found += [self.idioms[k]["idiom"] for k in sorted(hits) if self._matches(self.idioms[k], toks, positions)]
        return found


# ---------------------------------------------------------------- passages

def split_of(source, doc_id, eval_permille):
    h = int(hashlib.sha1(f"{source}/{doc_id}".encode()).hexdigest()[:12], 16)
    return "eval" if h % 1000 < eval_permille else "train"


def blocks_of(source, doc, wordset):
    """[(block id, [paragraphs], meta)] of a document: the turns of a debate, or the document itself"""
    if source == "handelingen":
        out = []
        for t, turn in enumerate(doc["turns"] or []):
            speaker = turn.get("speaker")
            if not speaker or SPEAKER_SKIP.search(speaker):
                continue
            paras = [p for p in (clean(x) for x in (turn.get("text") or "").split("\n")) if p]
            out.append((f"t{t}", paras, {"speaker": speaker, "party": turn.get("party")}))
        return out
    return [("d", paragraphs_of(source, doc["text"], wordset), {})]


class Builder:
    """Passages of documents, with every filter's verdict and the idiom score (one per process)"""

    def __init__(self, tokenizer_dir, wordlist, idioms):
        import langid
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
        self.langid = langid
        self.langid.classify("init")
        with open(wordlist, encoding="utf-8") as f:
            self.wordset = {line.strip().lower() for line in f if line.strip()}
        with open(idioms, encoding="utf-8") as f:
            self.matcher = IdiomMatcher(json.load(f))

    def count(self, texts):
        return [len(e.ids) for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)] if texts else []

    def pieces(self, paragraphs, labels, rng):
        """[(text, tokens, alone)] or None for a barrier: prose paragraphs, long ones cut into sentence runs"""
        counts = self.count(paragraphs)
        out = []
        for para, n, label in zip(paragraphs, counts, labels):
            if label:
                out.append(None)  # a new speaker: a passage never runs over it
            if not is_prose(para):
                out.append(None)
                continue
            if n <= MAX_TOKENS:
                out.append((para, n, False))
                continue
            sentences = split_sentences(para)
            sizes = self.count(sentences)
            run, total, target = [], 0, rng.randint(2 * MIN_TOKENS, MAX_TOKENS)
            for s, k in zip(sentences, sizes):
                if run and total + k > target:
                    if total >= MIN_TOKENS:
                        out.append((" ".join(run), total, True))
                    run, total, target = [], 0, rng.randint(2 * MIN_TOKENS, MAX_TOKENS)
                if k > MAX_TOKENS:
                    out.append(None)
                    continue
                run.append(s)
                total += k
            if run and total >= MIN_TOKENS:
                out.append((" ".join(run), total, True))
            out.append(None)
        return out

    def windows(self, pieces, rng):
        """Passages of 1-MAX_PARAGRAPHS consecutive pieces between barriers, as many paragraphs as drawn per passage"""
        i, out = 0, []
        while i < len(pieces):
            if pieces[i] is None:
                i += 1
                continue
            want = rng.randint(1, MAX_PARAGRAPHS)
            j, total = i, 0
            while j < len(pieces) and pieces[j] is not None and j - i < want:
                if pieces[j][2] and j > i:
                    break  # a cut paragraph stands alone
                if total + pieces[j][1] > MAX_TOKENS:
                    break
                total += pieces[j][1]
                j += 1
                if pieces[j - 1][2]:
                    break
            if j == i:  # a piece that does not fit with the one before
                i += 1
                continue
            if total >= MIN_TOKENS:
                out.append(([p[0] for p in pieces[i:j]], any(p[2] for p in pieces[i:j])))
            i = j
        return out

    def checks(self, source, text):
        """Every filter's verdict and the measures behind them"""
        lower = [w for w in words_of(text) if w.islower()]
        oov = sum(w not in self.wordset for w in lower) / max(len(lower), 1)
        chars = [c for c in text if not c.isspace()]
        digits = sum(c.isdigit() for c in chars) / max(len(chars), 1)
        punct = len(PUNCT.findall(text)) / max(len(chars), 1)
        words = words_of(text)
        capitalized = sum(w[0].isupper() for w in words) / max(len(words), 1)
        lang, _ = self.langid.classify(text)
        fails = []
        if lang != "nl":
            fails.append("langid")
        if any(w.lower() in OLD_SPELLING for w in words_of(text)):
            fails.append("old_spelling")
        if oov > MAX_OOV:
            fails.append("oov")
        if digits > MAX_DIGITS:
            fails.append("digits")
        if punct > MAX_PUNCT:
            fails.append("punct")
        if capitalized > MAX_CAPITALIZED:
            fails.append("capitals")
        if MOTION.search(text):
            fails.append("motion")
        if source != "handelingen" and name_conflict(text):
            fails.append("name_conflict")
        if source == "raad" and INLINE_LABEL.search(text):
            fails.append("speaker_change")
        return fails, {"oov": round(oov, 4), "digits": round(digits, 4), "punct": round(punct, 4),
                       "capitalized": round(capitalized, 4), "lang": lang}

    def document(self, source, doc):
        """(passages, counters) of one document"""
        counters = collections.Counter()
        doc_id = str(doc["id"] if source == "handelingen" else doc["doc_id"])
        rng = random.Random(f"{source}/{doc_id}")
        passages = []
        for block_id, paragraphs, meta in blocks_of(source, doc, self.wordset):
            labels = [False] * len(paragraphs)
            if source == "raad":
                for k, p in enumerate(paragraphs):
                    m = SPEAKER_LABEL.match(p)
                    if m:
                        labels[k] = True
                        paragraphs[k] = p[m.end():]
                counters["speaker_labels"] += sum(labels)
            counters["paragraphs"] += len(paragraphs)
            pieces = self.pieces(paragraphs, labels, rng)
            counters["prose_pieces"] += sum(p is not None for p in pieces)
            for k, (paras, cut) in enumerate(self.windows(pieces, rng)):
                text = "\n".join(paras)
                fails, measures = self.checks(source, text)
                hits = self.matcher.find(text)
                n_words = len(words_of(text))
                passages.append({
                    "pid": f"{source}:{doc_id}:{block_id}:{k}", "source": source, "doc_id": doc_id,
                    "text": text, "paragraphs": len(paras), "piece": cut, "fails": fails, **measures,
                    "words": n_words, "idioms": hits, "idiom_score": round(100 * len(hits) / max(n_words, 1), 4),
                    "speaker": meta.get("speaker"), "party": meta.get("party"),
                    "date": doc.get("date"), "content_type": list(doc.get("content_type") or []),
                })
        texts = [p["text"] for p in passages]
        for p, n in zip(passages, self.count(texts)):
            p["tokens"] = n
            if not MIN_TOKENS <= n <= MAX_TOKENS:
                p["fails"].append("length")  # the newlines between paragraphs can tip it over
        counters["passages"] += len(passages)
        return passages, counters


_BUILDER = None


def _init(tokenizer_dir, wordlist, idioms):
    global _BUILDER
    _BUILDER = Builder(tokenizer_dir, wordlist, idioms)


def _work(task):
    source, docs = task
    out, counters = [], collections.Counter()
    for doc in docs:
        passages, c = _BUILDER.document(source, doc)
        out += passages
        counters.update(c)
    return source, out, counters


# ---------------------------------------------------------------- reading

def row_groups(source, n, seed, fs):
    """n row groups of the source's files, spread evenly from a random offset: [(file, row group, rows)]"""
    import pyarrow.parquet as pq

    repo, revision = SOURCES[source]["repo"]
    groups = []
    for name in SOURCES[source]["files"]:
        with fs.open(f"datasets/{repo}@{revision}/{name}", "rb") as f:
            md = pq.ParquetFile(f).metadata
            groups += [(name, g, md.row_group(g).num_rows) for g in range(md.num_row_groups)]
    if n >= len(groups):
        return groups
    step = len(groups) / n
    offset = random.Random(f"{seed}/{source}").random() * step
    return [groups[int(offset + k * step)] for k in range(n)]


def read_docs(source, groups, fs):
    """The documents of the chosen row groups that pass the document filters, and the counts of those filters"""
    import pyarrow.parquet as pq

    repo, revision = SOURCES[source]["repo"]
    columns = SOURCES[source].get("columns", PROPELLA_COLUMNS)
    counters, docs = collections.Counter(), []
    by_file = collections.defaultdict(list)
    for name, g, _ in groups:
        by_file[name].append(g)
    for name, gs in by_file.items():
        with fs.open(f"datasets/{repo}@{revision}/{name}", "rb", block_size=1 << 23) as f:
            table = pq.ParquetFile(f).read_row_groups(sorted(gs), columns=columns)
        for doc in table.to_pylist():
            counters["docs_read"] += 1
            if source == "handelingen":
                if doc["chamber"] != "tweede_kamer":
                    counters["docs_other_chamber"] += 1
                    continue
            else:
                types = set(doc["content_type"] or [])
                if doc["content_quality"] not in GOOD_QUALITY:
                    counters["docs_quality"] += 1
                    continue
                if doc["content_integrity"] not in GOOD_INTEGRITY:
                    counters["docs_integrity"] += 1
                    continue
                if not types - NOT_CONTENT:
                    counters["docs_boilerplate_or_data"] += 1
                    continue
            docs.append(doc)
    counters["docs_kept"] = len(docs)
    return docs, counters


# ---------------------------------------------------------------- choosing

def ngrams(text, n=NGRAM):
    toks = [t.lower() for t in NGRAM_WORD.findall(text)]
    return {hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def normalized(text):
    return " ".join(NGRAM_WORD.findall(text.lower()))


class Chooser:
    """Draws passages one source at a time, keeping the exact and 13-gram duplicates and the per-document cap out"""

    def __init__(self, max_per_doc, blocked_ngrams=()):
        self.max_per_doc = max_per_doc
        self.seen_texts, self.seen_ngrams = set(), set(blocked_ngrams)
        self.per_doc = collections.Counter()
        self.skipped = collections.Counter()

    def take(self, candidates, n, rng, weight=lambda p: 1.0):
        """Up to n of the candidates, drawn without replacement with the given weights (Efraimidis-Spirakis keys)"""
        keyed = sorted(candidates, key=lambda p: -(rng.random() ** (1.0 / weight(p))))
        chosen = []
        for p in keyed:
            if len(chosen) == n:
                break
            if self.per_doc[(p["source"], p["doc_id"])] >= self.max_per_doc:
                self.skipped["per_doc_cap"] += 1
                continue
            norm = normalized(p["text"])
            if norm in self.seen_texts:
                self.skipped["exact_duplicate"] += 1
                continue
            grams = ngrams(p["text"])
            if grams & self.seen_ngrams:
                self.skipped["ngram_overlap"] += 1
                continue
            self.seen_texts.add(norm)
            self.seen_ngrams |= grams
            self.per_doc[(p["source"], p["doc_id"])] += 1
            chosen.append(p)
        return chosen


def quotas(n):
    """Passages per source for n in all, by SHARES, rounded so that they add up to n"""
    raw = {s: n * share for s, share in SHARES.items()}
    out = {s: math.floor(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - out[s], reverse=True)[:n - sum(out.values())]:
        out[s] += 1
    return out


def choose(pool, n, chooser, rng, source):
    """n passages of one source: half from the highest idiom class, half at random from all that pass"""
    weight = (lambda p: RAAD_PREFERRED_WEIGHT if set(p["content_type"]) & RAAD_PREFERRED else 1.0) \
        if source == "raad" else (lambda p: 1.0)
    n_idiom = n // 2
    ranked = sorted((p for p in pool if p["idiom_score"] > 0), key=lambda p: -p["idiom_score"])
    top = ranked[:IDIOM_CLASS_FACTOR * n_idiom]
    threshold = top[-1]["idiom_score"] if top else None
    top = [p for p in ranked if threshold is not None and p["idiom_score"] >= threshold]
    idiom_half = chooser.take(top, n_idiom, rng, weight)
    for p in idiom_half:
        p["half"] = "idiom"
    chosen_ids = {p["pid"] for p in idiom_half}
    random_half = chooser.take([p for p in pool if p["pid"] not in chosen_ids], n - len(idiom_half), rng, weight)
    for p in random_half:
        p["half"] = "random"
    return idiom_half + random_half, {"idiom_class_threshold": threshold, "idiom_class_size": len(top),
                                       "pool": len(pool), "pool_with_idiom": len(ranked)}


def drop_frequent_idioms(passages, max_share):
    """Idioms found in more than max_share of the passages that pass the filters leave every passage's idioms and
    score; returns [(idiom, passages)] of those left out, most frequent first"""
    ok = [p for p in passages if not p["fails"]]
    df = collections.Counter(i for p in ok for i in set(p["idioms"]))
    frequent = {i: n for i, n in df.items() if n > max_share * len(ok)}
    for p in passages:
        p["idioms"] = [i for i in p["idioms"] if i not in frequent]
        p["idiom_score"] = round(100 * len(p["idioms"]) / max(p["words"], 1), 4)
    return sorted(frequent.items(), key=lambda x: -x[1]), df


def percentiles(values, qs=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    if not values:
        return {}
    v = sorted(values)
    return {f"p{q}": v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] for q in qs}


def balanced_workers(passages, workers):
    """Worker k gets every workers-th passage in order of length, so each has a similar mix of lengths"""
    order = sorted(passages, key=lambda p: (p["tokens"], p["id"]))
    return [order[k::workers] for k in range(workers)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_per_source(text, default):
    out = dict.fromkeys(SOURCES, default)
    for part in filter(None, (text or "").split(",")):
        key, value = part.split("=")
        if key not in SOURCES:
            raise SystemExit(f"unknown source {key!r} in {text!r}")
        out[key] = int(value)
    return out


def select(args):
    import concurrent.futures
    import multiprocessing

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.parquet")):
        raise SystemExit(f"{out} has parquet files already: a new selection goes to a new directory")
    fs = HfFileSystem()
    n_groups = parse_per_source(args.row_groups, 0)
    max_docs = parse_per_source(args.max_docs, 0)
    report = {"args": vars(args), "sources": {}, "thresholds": {
        "min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS, "max_paragraphs": MAX_PARAGRAPHS, "max_oov": MAX_OOV,
        "max_digits": MAX_DIGITS, "max_punct": MAX_PUNCT, "max_capitalized": MAX_CAPITALIZED,
        "idiom_class_factor": IDIOM_CLASS_FACTOR,
        "raad_preferred_weight": RAAD_PREFERRED_WEIGHT, "ngram": NGRAM},
        "resources": {"wordlist": sha256(args.wordlist), "idioms": sha256(args.idioms)}}
    started = time.time()
    tasks, groups_read = [], {}
    for source in SOURCES:
        groups = row_groups(source, n_groups[source], args.seed, fs)
        groups_read[source] = [[name, g] for name, g, _ in groups]
        docs, counters = read_docs(source, groups, fs)
        if max_docs[source] and len(docs) > max_docs[source]:
            docs = random.Random(f"{args.seed}/{source}/docs").sample(docs, max_docs[source])
            counters["docs_sampled"] = len(docs)
        report["sources"][source] = {"row_groups": len(groups), "documents": dict(counters)}
        print(f"{source}: {len(groups)} row groups, {dict(counters)} ({time.time() - started:.0f} s)", flush=True)
        for k in range(0, len(docs), args.docs_per_task):
            tasks.append((source, docs[k:k + args.docs_per_task]))
    report["row_groups_read"] = groups_read

    passages = collections.defaultdict(list)
    counters = collections.defaultdict(collections.Counter)
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(args.processes, mp_context=ctx, initializer=_init,
                                                initargs=(args.tokenizer, args.wordlist, args.idioms)) as pool:
        for source, found, c in pool.map(_work, tasks):
            passages[source] += found
            counters[source].update(c)
    print(f"passages built ({time.time() - started:.0f} s): "
          f"{ {s: len(v) for s, v in passages.items()} }", flush=True)
    frequent, df = drop_frequent_idioms([p for v in passages.values() for p in v], args.max_idiom_share)
    kept = [(i, n) for i, n in df.most_common() if i not in dict(frequent)]
    report["idioms"] = {"max_share": args.max_idiom_share, "left_out": len(frequent), "kept_found": len(kept),
                        "left_out_list": frequent, "kept_most_frequent": kept[:60]}
    print(f"idioms: {len(frequent)} too frequent, left out of the score; {len(kept)} others found", flush=True)

    rng = random.Random(args.seed)
    want, want_eval = quotas(args.n), quotas(args.n_eval) if args.n_eval else dict.fromkeys(SOURCES, 0)
    blocked = set()
    for path in args.block_texts or []:
        with open(path, encoding="utf-8") as f:
            for line in f:
                blocked |= ngrams(line)
    chooser = Chooser(args.max_per_doc, blocked)
    chosen, chosen_eval, rejected = [], [], []
    for source in SOURCES:
        allp = passages[source]
        fails = collections.Counter(f for p in allp for f in p["fails"])
        first = collections.Counter(p["fails"][0] for p in allp if p["fails"])
        for p in allp:
            p["split"] = split_of(source, p["doc_id"], args.eval_permille)
        ok = [p for p in allp if not p["fails"]]
        train = [p for p in ok if p["split"] == "train"]
        evalp = [p for p in ok if p["split"] == "eval"]
        picked, info = choose(train, want[source], chooser, rng, source)
        if len(picked) < want[source]:
            raise SystemExit(f"{source}: only {len(picked)} passages for {want[source]} (pool {len(train)}): "
                             f"read more row groups")
        picked_eval, info_eval = choose(evalp, want_eval[source], chooser, rng, source) if want_eval[source] \
            else ([], {})
        if len(picked_eval) < want_eval[source]:
            raise SystemExit(f"{source}: only {len(picked_eval)} eval passages for {want_eval[source]}")
        chosen += picked
        chosen_eval += picked_eval
        r = random.Random(f"{args.seed}/{source}/rejected")
        for name in sorted(fails):
            failing = [p for p in allp if name in p["fails"]]
            rejected += r.sample(failing, min(args.rejected_per_filter, len(failing)))
        report["sources"][source].update({
            "counts": dict(counters[source]), "fails": dict(fails), "first_fail": dict(first),
            "passed": len(ok), "passed_train": len(train), "passed_eval": len(evalp), "chosen": len(picked),
            "chosen_eval": len(picked_eval), "choice": info, "choice_eval": info_eval,
            "distributions_all": {m: percentiles([p[m] for p in allp]) for m in
                                  ("oov", "digits", "punct", "capitalized")},
            "distributions_passed": {m: percentiles([p[m] for p in train]) for m in
                                     ("tokens", "idiom_score", "paragraphs")},
            "chosen_tokens": percentiles([p["tokens"] for p in picked]),
            "chosen_piece": sum(p["piece"] for p in picked),
            "chosen_halves": dict(collections.Counter(p["half"] for p in picked)),
            "chosen_with_idiom": sum(bool(p["idioms"]) for p in picked),
            "chosen_preferred_types": sum(bool(set(p["content_type"]) & RAAD_PREFERRED) for p in picked),
            "chosen_starts_voorzitter": sum(p["text"].startswith("Voorzitter") for p in picked),
        })
    report["skipped_while_choosing"] = dict(chooser.skipped)
    for k, p in enumerate(chosen):
        p["id"] = k
    for k, p in enumerate(chosen_eval):
        p["id"] = 1_000_000 + k
    eval_docs = {(p["source"], p["doc_id"]) for p in chosen_eval}
    if any((p["source"], p["doc_id"]) in eval_docs for p in chosen):
        raise SystemExit("a document has both training and eval passages")

    columns = ["id", "pid", "source", "doc_id", "split", "half", "text", "tokens", "paragraphs", "piece", "words",
               "idiom_score", "idioms", "oov", "digits", "punct", "capitalized", "lang", "speaker", "party", "date",
               "content_type"]

    def write(rows, name):
        pq.write_table(pa.table({c: [r.get(c) for r in rows] for c in columns}), out / name, compression="zstd")

    write(chosen, "passages.parquet")
    if chosen_eval:
        write(chosen_eval, "eval.parquet")
    pq.write_table(pa.table({c: [r.get(c) for r in rejected] for c in columns + ["fails"]}), out / "rejected.parquet")
    for k, part in enumerate(balanced_workers(chosen, args.workers)):
        pq.write_table(pa.table({"id": [p["id"] for p in part], "text": [p["text"] for p in part],
                                 "tokens": [p["tokens"] for p in part]}), out / f"segments-w{k}.parquet")
    report["chosen"] = len(chosen)
    report["chosen_eval"] = len(chosen_eval)
    report["seconds"] = round(time.time() - started)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"SELECT {json.dumps({s: report['sources'][s]['chosen'] for s in SOURCES})} in {report['seconds']} s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("idioms", help="The idiom list from kaikki.org's Dutch extract")
    p.add_argument("--kaikki", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("select", help="Choose the passages")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True, help="Training-side passages (2,000 for the pilot)")
    p.add_argument("--n-eval", type=int, default=0, help="Eval passages, from other documents (4,000 later)")
    p.add_argument("--eval-permille", type=int, default=20, help="Documents on the eval side, per mille")
    p.add_argument("--idioms", required=True)
    p.add_argument("--wordlist", required=True)
    p.add_argument("--tokenizer", required=True, help="TranslateGemma's tokenizer directory (tokenizer.json)")
    p.add_argument("--row-groups", default="handelingen=8,raad=12,dpc=8,rechtspraak=6",
                   help="Row groups to read per source, spread over its files")
    p.add_argument("--max-docs", default="", help="At most this many documents per source (random), e.g. raad=4000")
    p.add_argument("--max-per-doc", type=int, default=3)
    p.add_argument("--max-idiom-share", type=float, default=MAX_IDIOM_SHARE,
                   help="Idioms in more of the passing passages than this share do not count")
    p.add_argument("--block-texts", nargs="*", help="Text files (a text a line) whose 13-grams no passage may share")
    p.add_argument("--workers", type=int, default=4, help="Segment files for translate.py")
    p.add_argument("--processes", type=int, default=32)
    p.add_argument("--docs-per-task", type=int, default=50)
    p.add_argument("--rejected-per-filter", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "idioms":
        with open(args.kaikki, encoding="utf-8") as f:
            idioms = build_idioms(f)
        Path(args.out).write_text(json.dumps(idioms, ensure_ascii=False))
        print(f"{len(idioms)} idioms, {collections.Counter(i['pos'] for i in idioms)}")
        return
    select(args)


if __name__ == "__main__":
    sys.exit(main())
