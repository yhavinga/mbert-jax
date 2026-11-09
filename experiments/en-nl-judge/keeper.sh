#!/bin/bash
# On co1: keeps the judge's spot VM going until run.sh has ended, through hammond's TPU guard only
# (training/tpu.py up, with --replace for a preempted VM of this name). tpu.py watch cannot do it: it deploys and
# starts hammond's trainer. Each cycle reads the VM's state and takes at most one action:
#   no request             tpu.py up
#   VM preempted/stopped,  tpu.py up --replace (refused while the old request is still SUSPENDING: tried again next
#   request failed         cycle)
#   VM READY, no run log   HF token through stdin, scripts from the bucket of the region, run.sh detached
#   VM READY, run log      its last lines; EXIT 0 ends the keeper (exit 0), another EXIT too (exit 2); the VM stays
# Start: setsid nohup bash keeper.sh > ~/logs/judge_keeper.log 2>&1 < /dev/null &
set -uo pipefail
NAME=${NAME:-en-nl-judge-use1}
ZONE=${ZONE:-us-east1-d}
case ${ZONE%-*} in
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  *) echo "no judge bucket for zone '$ZONE'"; exit 1 ;;
esac
SCRIPTS=$BUCKET/mbert/translator/judge/scripts
HAMMOND=$HOME/Developer/yhavinga/hammond
POLL=${POLL:-60}
log() { echo "$(date -u +%FT%TZ) $*"; }
ssh_vm() {
  # 180 s, not 900: a VM that is preempted between the state check and the ssh would hold the keeper for 15 minutes
  timeout 180 gcloud compute tpus tpu-vm ssh "$NAME" --zone "$ZONE" --project t5-dutch-english --quiet \
    --ssh-flag="-o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=4" --command "$1"
}
state() {
  (cd "$HAMMOND" && .venv/bin/python training/tpu.py status --json 2> /dev/null) | python3 -c "
import json, sys
d = json.load(sys.stdin)
z = d['zones']['$ZONE']
if not z['ok']:
    print('read_failed'); sys.exit()
qr = [q['state'] for q in z['queued_resources'] if q['name'] == '$NAME']
vm = [v['state'] for v in z['vms'] if v['name'] == '$NAME']
print((qr or ['none'])[0], (vm or ['none'])[0])"
}
up() {
  (cd "$HAMMOND" && .venv/bin/python training/tpu.py up --accelerator v6e-8 --zone "$ZONE" --name "$NAME" "$@" 2>&1 | tail -n 2)
}
last=""
while true; do
  read -r qr vm < <(state || echo "read_failed read_failed")
  now="qr=$qr vm=$vm"
  [ "$now" != "$last" ] && log "$now"
  last=$now
  case "$qr $vm" in
    read_failed*) ;;
    "none none") log "requesting"; up ;;
    *" PREEMPTED"|*" TERMINATED"|*" STOPPED"|"FAILED "*|"SUSPENDED "*) log "re-requesting"; up --replace ;;
    *" READY")
      # A run log without EXIT and without a run.sh process: the VM rebooted (a maintenance event empties /dev/shm);
      # the old log is moved aside and run.sh starts again
      out=$(ssh_vm 'if [ -f ~/logs/judge_run.log ] && ! tail -n 1 ~/logs/judge_run.log | grep -q "^EXIT" && ! pgrep -f "[r]un.sh" > /dev/null; then mv ~/logs/judge_run.log ~/logs/judge_run.$(date +%s).log; fi; if [ -f ~/logs/judge_run.log ]; then echo RUNLOG; tail -n 3 ~/logs/judge_run.log | cut -c1-300; else echo NORUN; fi' < /dev/null 2> /dev/null)
      if echo "$out" | grep -q '^NORUN'; then
        log "new VM: token, scripts, run.sh"
        ssh_vm 'mkdir -p ~/.cache/huggingface && umask 077 && cat > ~/.cache/huggingface/token' < "$HOME/.cache/huggingface/token" > /dev/null 2>&1 \
          && ssh_vm "mkdir -p /dev/shm/judge/code ~/logs && gsutil -q cp '$SCRIPTS/*' /dev/shm/judge/code/ && (setsid nohup bash -c 'bash /dev/shm/judge/code/run.sh > ~/logs/judge_run.log 2>&1; echo EXIT \$? >> ~/logs/judge_run.log' > /dev/null 2>&1 < /dev/null &) && echo started" < /dev/null 2>&1 | tail -n 1
      elif echo "$out" | grep -q '^RUNLOG'; then
        tail_line=$(echo "$out" | grep -v '^RUNLOG' | tail -n 1)
        log "run: $tail_line"
        if echo "$out" | grep -q '^EXIT 0$'; then log "run.sh ended in EXIT 0"; exit 0; fi
        if echo "$out" | grep -q '^EXIT [1-9]'; then log "run.sh failed: $(echo "$out" | tr '\n' ' ')"; exit 2; fi
      fi
      ;;
  esac
  sleep "$POLL"
done
