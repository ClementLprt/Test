#!/bin/bash
# Lance Intercom Live.
#  - à la main (une seule fois) :  bash ~/Downloads/intercom-live/lancer.command
#    → installe ce qu'il faut et crée l'app « Intercom Live » dans le dossier Applications de ta session.
#  - ensuite : l'app (Launchpad, Dock, Spotlight) appelle ce script avec --app, sans Terminal.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"   # l'app lancée du Dock a un PATH minimal
APPDIR="$(pwd)"
APPMODE=0; [ "$1" = "--app" ] && { APPMODE=1; shift; }
xattr -dr com.apple.quarantine . 2>/dev/null
chmod +x lancer.command 2>/dev/null

notify() { osascript -e "display notification \"$1\" with title \"Intercom Live\"" >/dev/null 2>&1; }
pause() { [ $APPMODE = 1 ] && return; echo; read -r -p "Appuie sur Entrée pour fermer…" _; }
fail() {
  echo "❌ $1"
  if [ $APPMODE = 1 ]; then
    osascript -e "display alert \"Intercom Live\" message \"$1\n\nDétail : $APPDIR/dernier_lancement.log\"" >/dev/null 2>&1
  fi
  pause; exit 1
}

# En mode app, tout va dans le journal (pas de fenêtre Terminal).
if [ $APPMODE = 1 ]; then exec >>dernier_lancement.log 2>&1; echo "=== $(date) ==="; fi

# 1. Trouver un Python entre 3.11 et 3.13 (Apple = 3.9 trop vieux ; 3.14+ trop récent pour certains modules).
find_python() {
  for c in python3.12 python3.13 python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    p=$(command -v "$c" 2>/dev/null) || continue
    if "$p" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then echo "$p"; return; fi
  done
}

pip_install() {   # venv classique (pip) ou créé par uv (sans pip)
  if .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/python -m pip install -r requirements.txt
  else
    UV=$(command -v uv || echo "$HOME/.local/bin/uv")
    "$UV" pip install --python .venv/bin/python -r requirements.txt
  fi
}

install_venv() {
  PY=$(find_python)
  [ $APPMODE = 1 ] && notify "Installation des modules (quelques minutes)…"
  if [ -n "$PY" ]; then
    echo "Installation avec $PY ($("$PY" --version))…"
    "$PY" -m venv .venv && pip_install \
      || { rm -rf .venv; fail "L'installation des modules a échoué."; }
  else
    if [ $APPMODE = 0 ]; then
      echo "Il faut Python 3.11, 3.12 ou 3.13. Trouvé : $(python3 --version 2>&1)."
      echo "J'installe Python 3.12 rien que pour cette app, avec l'outil « uv » (sans mot de passe)."
      read -r -p "OK ? (o/n) " rep
      case "$rep" in o|O|oui|y|Y) ;; *) fail "Installe Python 3.12 depuis https://www.python.org/downloads/macos/ puis relance.";; esac
    fi
    UV=$(command -v uv || echo "$HOME/.local/bin/uv")
    if [ ! -x "$UV" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | sh || fail "Installation de uv impossible (réseau ?)."
      UV="$HOME/.local/bin/uv"
    fi
    "$UV" venv --python 3.12 .venv || fail "Installation de Python impossible."
    "$UV" pip install --python .venv/bin/python -r requirements.txt || { rm -rf .venv; fail "Installation des modules impossible."; }
  fi
  shasum requirements.txt | cut -d' ' -f1 > .venv/.req_hash
}

[ -x .venv/bin/python ] || install_venv

# Lancement à la main sur Mac : on (re)crée l'app « Intercom Live » et on l'ouvre. Plus besoin du Terminal ensuite.
if [ $APPMODE = 0 ] && [ "$(uname)" = "Darwin" ] && [ "$1" != "--console" ] && [ "$1" != "--file" ] && [ "$1" != "--list-devices" ]; then
  .venv/bin/python intercom_live.py --install-app || fail "Création de l'app impossible."
  echo; echo "✅ L'app « Intercom Live » est installée dans ton dossier Applications (Launchpad / Spotlight)."
  echo "   Glisse-la dans le Dock si tu veux."
  open "$HOME/Applications/Intercom Live.app"
  echo "Démarrage de l'app…"
  for i in $(seq 1 40); do
    sleep 1
    if curl -s -m 1 http://127.0.0.1:8765/config >/dev/null 2>&1; then
      echo "✅ L'app tourne. Tu peux fermer cette fenêtre (Cmd+Q sur Terminal)."
      exit 0
    fi
  done
  echo; echo "⚠ L'app ne répond pas. Voici son journal :"; tail -n 30 dernier_lancement.log 2>/dev/null
  echo; echo "Je la lance dans cette fenêtre à la place (laisse le Terminal ouvert)…"
  pkill -f "intercom_live.py" 2>/dev/null; sleep 1
fi
[ "$1" = "--console" ] && shift

trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM INT   # « Quitter » depuis le Dock arrête aussi le logiciel
EXTRA=""

# 2. Boucle : le logiciel se relance tout seul quand son code est mis à jour (code de sortie 42).
while true; do
  # nouveaux modules demandés par une mise à jour ?
  if [ "$(shasum requirements.txt | cut -d' ' -f1)" != "$(cat .venv/.req_hash 2>/dev/null)" ]; then
    echo "Mise à jour des modules…"
    pip_install && shasum requirements.txt | cut -d' ' -f1 > .venv/.req_hash
  fi
  if [ $APPMODE = 1 ]; then
    .venv/bin/python -u intercom_live.py --app $EXTRA "$@" &
    wait $!
    code=$?
  else
    .venv/bin/python -u intercom_live.py $EXTRA "$@" 2>&1 | tee dernier_lancement.log
    code=${PIPESTATUS[0]}
  fi
  [ "$code" = "42" ] && { echo "↻ Mise à jour détectée, redémarrage…"; EXTRA="--no-browser"; sleep 1; continue; }
  break
done
echo; echo "Le logiciel s'est arrêté (code $code)."
[ "$code" != "0" ] && [ $APPMODE = 1 ] && fail "Le logiciel s'est arrêté sur une erreur."
pause
