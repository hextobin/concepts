#!/bin/bash
# Run every 5 minutes by a systemd timer, independent of training. Self-destructs
# when the deadline in /root/deadline has passed, or when control/stop has been
# pushed to main (the remote kill switch).
source /root/.concepts.env
if [ "$(date +%s)" -ge "$(cat /root/deadline)" ]; then
  exec /root/concepts/box/self_destruct.sh "deadline passed"
fi
cd /root/concepts
if git fetch -q origin main && git cat-file -e origin/main:control/stop 2>/dev/null; then
  exec /root/concepts/box/self_destruct.sh "control/stop pushed to main"
fi
