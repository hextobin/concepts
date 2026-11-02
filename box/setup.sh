#!/bin/bash
# First boot on the droplet (run by cloud-init as root): arm the dead-man timer,
# schedule result syncs, and start the grid.
set -euo pipefail
source /root/.concepts.env
mkdir -p /root/logs
echo $(( $(date +%s) + DEADLINE_HOURS * 3600 )) > /root/deadline
chmod +x /root/concepts/box/*.sh

unit() { cat > "/etc/systemd/system/$1"; }

unit concepts-deadman.service <<'EOF'
[Service]
Type=oneshot
TimeoutStartSec=infinity
ExecStart=/root/concepts/box/deadman.sh
EOF
unit concepts-deadman.timer <<'EOF'
[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
EOF

unit concepts-sync.service <<'EOF'
[Service]
Type=oneshot
TimeoutStartSec=10min
ExecStart=/root/concepts/box/sync.sh
EOF
unit concepts-sync.timer <<'EOF'
[Timer]
OnBootSec=15min
OnUnitActiveSec=15min
[Install]
WantedBy=timers.target
EOF

unit concepts-grid.service <<'EOF'
[Service]
Type=simple
ExecStart=/root/concepts/box/run_grid.sh
EOF

systemctl daemon-reload
systemctl enable --now concepts-deadman.timer concepts-sync.timer
systemctl start --no-block concepts-grid.service
