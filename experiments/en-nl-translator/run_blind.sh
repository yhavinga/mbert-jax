#!/bin/bash
# The passages of the blind assessment (blind.py; protocol.md §4.2.4), on a worker of mbert-mix: CPU and the internet,
# no TPU. Data from the internet goes straight to the worker and to the bucket of us-central2. Each stage runs detached,
# log ~/logs/<job>.log ending in "EXIT <code>".
#   bash run_blind.sh prepare    a worker: a venv (spaCy, en_core_web_sm, Beautiful Soup), kaikki.org's English extract
#                                of Wiktionary, the English posts of Global Voices of 2025 (WordPress API), the idiom
#                                list and the candidates, to $S/blind/source/                  (job blind_prepare_$TAG)
#   SETS="dir ..." bash run_blind.sh select   the worker that holds the training sets (w3: the arms' build output):
#                                blind.py select, to $S/blind/; EXCLUDE="pid ..." leaves passages out (job blind_select_$TAG)
set -euo pipefail
STAGE=${1:?prepare or select}
S=gs://t5-dutch-english-usc2/mbert/translator/sft
M=gs://t5-dutch-english-usc2/mbert/translator/mainset
E=gs://t5-dutch-english-usc2/mbert/translator/eval
D=/dev/shm/translator/blind
PY=/dev/shm/translator/venv/bin/python  # the translator's venv: pyarrow
CODE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-manual}
W=$(hostname | sed 's/.*-w-//')
KAIKKI_URL=https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl
SPACY=spacy==3.8.7
SPACY_MODEL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
BS4=beautifulsoup4==4.13.4
CLICK=click==8.1.8  # spacy.cli imports click; the typer that spaCy 3.8.7 pulls in no longer brings it
AFTER=2025-01-01T00:00:00
BEFORE=2026-01-01T00:00:00
mkdir -p "$D" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

case $STAGE in
prepare)
  cat > ~/job_body_blind_prepare_$TAG.sh <<EOF
set -euo pipefail
cd $D
mkdir -p res src
[ -x $D/venv/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 $D/venv
~/.local/bin/uv pip install -q --python $D/venv/bin/python $SPACY $CLICK $BS4 $SPACY_MODEL
$D/venv/bin/python -c "import spacy, bs4; spacy.load('en_core_web_sm')"
~/.local/bin/uv pip freeze --python $D/venv/bin/python > src/venv-freeze.txt
if [ ! -s res/kaikki-English.jsonl ]; then
  modified=\$(curl -fsSI $KAIKKI_URL | tr -d '\r' | sed -n 's/^[Ll]ast-[Mm]odified: //p')
  curl -fsSL $KAIKKI_URL -o res/kaikki-English.jsonl.part
  mv res/kaikki-English.jsonl.part res/kaikki-English.jsonl
  echo "kaikki-English.jsonl $KAIKKI_URL last-modified '\$modified' sha256 \$(sha256sum res/kaikki-English.jsonl | cut -c1-64)" > res/kaikki.manifest
fi
[ -s res/kaikki.manifest ] || { echo "no res/kaikki.manifest beside res/kaikki-English.jsonl"; exit 1; }
cp res/kaikki.manifest src/manifest.txt
$D/venv/bin/python -I -u $CODE/blind.py idioms --kaikki res/kaikki-English.jsonl --out src/idioms.jsonl
$D/venv/bin/python -I -u $CODE/blind.py gv-fetch --after $AFTER --before $BEFORE --out src/gv-posts.jsonl
echo "gv-posts.jsonl https://globalvoices.org/wp-json/wp/v2/posts after $AFTER before $BEFORE fetched \$(date -u +%FT%TZ) sha256 \$(sha256sum src/gv-posts.jsonl | cut -c1-64)" >> src/manifest.txt
echo "spaCy: \$(grep -E '^(spacy|en-core-web-sm|beautifulsoup4)==' src/venv-freeze.txt | tr '\n' ' ')" >> src/manifest.txt
$D/venv/bin/python -I -u $CODE/blind.py candidates --posts src/gv-posts.jsonl --idioms src/idioms.jsonl --out src/candidates.jsonl
gsutil -m -q cp src/* $S/blind/source/
echo "PREPARED: $S/blind/source/"
EOF
  detach "blind_prepare_$TAG" bash ~/job_body_blind_prepare_$TAG.sh
  ;;
select)
  SETS=${SETS:?the training set directories, e.g. /dev/shm/translator/sft/out-arms-v2-<tag>/N}
  O=$D/select-$TAG
  cat > ~/job_body_blind_select_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/in $O
gsutil -m -q cp $S/blind/source/candidates.jsonl $S/blind/source/candidates.jsonl.stats.json $S/blind/source/manifest.txt $D/in/
gsutil -q cp $M/flan-eval/eval-rows.parquet $E/data/evalsets.parquet $S/tagtest/segments/items.parquet $D/in/
$PY -I -u $CODE/blind.py select --candidates $D/in/candidates.jsonl --flan-eval $D/in/eval-rows.parquet --evalsets $D/in/evalsets.parquet --tagtest-items $D/in/items.parquet --train $SETS --manifest $D/in/manifest.txt --exclude ${EXCLUDE:-} --out $O
gsutil -m -q cp $O/idiom30.jsonl $O/blind100.jsonl $O/select-report.json $O/blind-idiom-30.md $S/blind/
echo "SELECTED: $S/blind/"
EOF
  detach "blind_select_$TAG" bash ~/job_body_blind_select_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
