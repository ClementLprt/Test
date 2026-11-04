#!/bin/bash
# Installation d'Intercom Live en une commande (à coller une seule fois dans Terminal) :
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/ClementLprt/Test/main/install.sh)"
set -e
REPO="ClementLprt/Test"
DEST="$HOME/Documents/Intercom Live/app"
echo "Installation d'Intercom Live dans : $DEST"
# arrête une version déjà ouverte (ancienne installation comprise)
curl -s -m 2 -X POST http://127.0.0.1:8765/quit -d '{}' >/dev/null 2>&1 || true
pkill -f "intercom_live.py" 2>/dev/null || true
sleep 1
mkdir -p "$DEST"
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/main" | tar -xz -C "$DEST" --strip-components=1
chmod +x "$DEST/lancer.command"
exec bash "$DEST/lancer.command"
