#!/usr/bin/env bash
# deploy.sh — assistente (gflights-assistente.service, porta 8006)
set -euo pipefail

VPS_HOST="178.104.179.185"
VPS_USER="gflights"
VPS_KEY="$HOME/.ssh/hetzner-gflights/id_ed25519"
APP_DIR_VPS="~/app-assistente"
SERVICE="gflights-assistente"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn(){ echo -e "${YELLOW}[warn]${NC} $*"; }
die(){ echo -e "${RED}[erro]${NC} $*" >&2; exit 1; }

cd "$(dirname "$0")"

if [[ "${1:-}" == "-m" && -n "${2:-}" ]]; then
    git add -A
    if git diff --cached --quiet; then warn "Nada pra commitar."
    else git commit -m "$2"; fi
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Mudanças não commitadas. Use './deploy.sh -m \"msg\"'"
    git status --short
    die "Abortando."
fi

log "Push pro GitHub..."
git push origin main

log "Pull na VPS e restart..."
ssh -i "$VPS_KEY" "$VPS_USER@$VPS_HOST" bash <<SSH_EOF
set -e
cd $APP_DIR_VPS
echo "[VPS] Pull..."
git pull --ff-only origin main
echo "[VPS] Restart $SERVICE..."
sudo systemctl restart $SERVICE
sleep 3
echo "[VPS] Status:"
systemctl is-active $SERVICE
SSH_EOF

log "Deploy OK"
