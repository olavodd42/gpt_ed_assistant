#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
: "${PHYSIONET_USERNAME:?Defina PHYSIONET_USERNAME no ambiente}"
MIMIC_ED_VERSION="${MIMIC_ED_VERSION:-1.0}"

DEST="${ROOT_DIR}/data/raw/mimic-iv-ed/${MIMIC_ED_VERSION}"
mkdir -p "$DEST"

# Baixa todos os arquivos do release escolhido; --ask-password evita expor senha
wget -r -N -c -np -nH --cut-dirs=3 \
  --user "$PHYSIONET_USERNAME" --ask-password \
  "https://physionet.org/files/mimic-iv-ed/${MIMIC_ED_VERSION}/" \
  -P "$DEST"
