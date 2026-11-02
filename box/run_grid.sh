#!/bin/bash
# On the droplet: check the GPU, prepare the full dataset, run the 14-run grid
# with an arc test after each run. Seed 0 of every variant runs first, so a
# partial grid is still a complete comparison. Log: /root/logs/grid.log
set -uo pipefail
cd /root/concepts
mkdir -p /root/logs
exec >>/root/logs/grid.log 2>&1
note() { echo "[$(date -u +%H:%M:%S)] $*"; }

note "installing python packages"
# First boot often has unattended upgrades holding the apt lock; wait for it.
apt-get -qq -o DPkg::Lock::Timeout=900 update
apt-get -qq -o DPkg::Lock::Timeout=900 install -y python3-venv >/dev/null
if ! { python3 -m venv .venv && .venv/bin/pip -q install torch numpy tokenizers; }; then
  exec /root/concepts/box/self_destruct.sh "python setup failed"
fi
if ! .venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  note "no GPU visible to torch"
  nvidia-smi || true
  exec /root/concepts/box/self_destruct.sh "no GPU visible"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

note "downloading TinyStories"
B=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
mkdir -p data
curl -sSL -o data/train.txt "$B/TinyStoriesV2-GPT4-train.txt"
curl -sSL -o data/valid.txt "$B/TinyStoriesV2-GPT4-valid.txt"
.venv/bin/python prep.py --train train.txt --show 0

run() {
  local name=$1; shift
  [ -f "runs/$name/arc.json" ] && return  # finished before a restart
  note "start $name"
  if ! .venv/bin/python train.py --name "$name" "$@"; then note "FAILED $name"; return; fi
  .venv/bin/python eval_arc.py "runs/$name" || note "arc test failed for $name"
  note "done $name"
}

for seed in 0 1; do
  s="--seed $seed"
  run "base-s$seed"              --mode baseline $s
  run "plan-clause-256-s$seed"   --mode plan --level clause --codes 256 --free-bits 2 $s
  run "summary-clause-256-s$seed" --mode summary --level clause --codes 256 $s
  run "plan-paragraph-256-s$seed" --mode plan --level paragraph --codes 256 --free-bits 2 $s
  run "plan-random-256-s$seed"   --mode plan --level random --codes 256 --free-bits 2 $s
  run "plan-clause-64-s$seed"    --mode plan --level clause --codes 64 --free-bits 2 $s
  run "plan-clause-1024-s$seed"  --mode plan --level clause --codes 1024 --free-bits 2 $s
done

note "grid finished"
touch /root/logs/GRID_DONE
/root/concepts/box/sync.sh
# Leave an hour for the managing agent to check results, then the timer fires.
echo $(( $(date +%s) + 3600 )) > /root/deadline
note "deadline moved to one hour from now"
