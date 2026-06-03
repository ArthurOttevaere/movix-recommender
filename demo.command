#!/usr/bin/env bash
#
# demo.sh — lance Movix pour la démo live (backend + site web).
#   Double-clic, ou depuis un terminal : ./demo.sh
#   Arrêter le serveur : Ctrl + C dans cette fenêtre.
#
set -e

# Se placer à la racine du projet (là où vit ce script), quel que soit
# le dossier depuis lequel on l'a lancé.
cd "$(dirname "$0")"

echo "=============================================="
echo "  Movix — démarrage de la démo"
echo "=============================================="

# 1) Activer l'environnement Python
if [ ! -d "venv" ]; then
  echo "ERREUR : dossier venv introuvable. Lance d'abord l'installation (voir README)."
  exit 1
fi
source venv/bin/activate
echo "[ok] environnement Python activé"

# 2) Afficher la branche git (info seulement — la démo marche sur n'importe
#    quelle branche puisque venv/artifacts/clés sont en dehors de git).
BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
echo "[info] branche git : $BRANCH"

# 3) Ouvrir le navigateur automatiquement (après un petit délai, le temps que
#    le serveur démarre). Sur macOS : commande 'open'.
URL="http://127.0.0.1:8000"
( sleep 3; open "$URL" >/dev/null 2>&1 || true ) &

echo "[ok] le site va s'ouvrir sur $URL"
echo "----------------------------------------------"
echo "  Choisis le profil \"Lenny\" pour une démo immédiate."
echo "  Pour arrêter : Ctrl + C ici."
echo "----------------------------------------------"

# 4) Lancer le serveur (sans --reload : plus stable pour une démo).
exec uvicorn backend.main:app
