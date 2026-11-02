#!/bin/bash
# Push logs, metrics and every checkpoint to the `results` branch.
# Runs every 15 minutes and once more before self-destruct.
set -uo pipefail
source /root/.concepts.env
R=/root/results
URL="https://x-access-token:${GH_TOKEN}@github.com/hextobin/concepts.git"

if [ ! -d "$R/.git" ]; then
  git clone -q --branch results "$URL" "$R" 2>/dev/null || {
    rm -rf "$R" && mkdir -p "$R"
    git -C "$R" init -q -b results && git -C "$R" remote add origin "$URL"
  }
fi

cd /root/concepts
mkdir -p "$R/logs" "$R/data" "$R/checkpoints"
cp -f /root/logs/*.log "$R/logs/" 2>/dev/null
cp -f /root/logs/GRID_DONE "$R/logs/" 2>/dev/null
cp -f data/meta.json data/tokenizer.json "$R/data/" 2>/dev/null
for d in runs/*/; do
  n=$(basename "$d")
  mkdir -p "$R/runs/$n"
  cp -f "$d"args.json "$d"log.jsonl "$d"arc.json "$R/runs/$n/" 2>/dev/null
done
for d in runs/*/; do
  n=$(basename "$d")
  if [ -f "runs/$n/model.pt" ] && [ ! -f "$R/checkpoints/$n.pt.gz" ]; then
    # bf16 halves the size so each checkpoint stays under GitHub's 100MB limit.
    .venv/bin/python - "runs/$n/model.pt" "/root/$n.pt" <<'EOF' && gzip -9 -c "/root/$n.pt" > "$R/checkpoints/$n.pt.gz"
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
ck["model"] = {k: v.bfloat16() if v.is_floating_point() else v for k, v in ck["model"].items()}
torch.save(ck, sys.argv[2])
EOF
    rm -f "/root/$n.pt"
  fi
done

cd "$R"
git add -A
git -c user.name=concepts-box -c user.email=box@concepts.invalid commit -qm "results $(date -u +%FT%TZ)" || true
git push -q origin results
