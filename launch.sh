#!/bin/bash
# Run on your Mac: creates the GPU droplet that runs the whole grid and then
# destroys itself. Tokens are typed at the prompt, only go into the droplet's
# startup script, and the temporary copy here is deleted on exit.
#
#   ./launch.sh [deadline_hours]   (default 10)
set -euo pipefail
HOURS=${1:-10}
cd "$(dirname "$0")"

git fetch -q origin main
if git cat-file -e origin/main:control/stop 2>/dev/null; then
  echo "control/stop is still on main (it would destroy the box at once). Remove it first." >&2
  exit 1
fi

read -rsp "DigitalOcean token (droplet read + delete): " DO_TOKEN; echo
read -rsp "GitHub token (concepts contents read/write): " GH_TOKEN; echo

UD=$(mktemp)
trap 'rm -f "$UD"' EXIT
chmod 600 "$UD"
cat > "$UD" <<EOF
#!/bin/bash
set -euo pipefail
umask 077
cat > /root/.concepts.env <<'ENV'
DO_TOKEN=$DO_TOKEN
GH_TOKEN=$GH_TOKEN
DEADLINE_HOURS=$HOURS
ENV
git clone -q --single-branch --branch main https://x-access-token:$GH_TOKEN@github.com/hextobin/concepts.git /root/concepts
bash /root/concepts/box/setup.sh
EOF

doctl compute droplet create concepts-run \
  --size gpu-6000adax1-48gb --region tor1 --image gpu-h100x1-base \
  --ssh-keys 60418976 --user-data-file "$UD" --wait \
  --format ID,Name,PublicIPv4,Status
