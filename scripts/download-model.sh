#!/usr/bin/env bash
# download-model.sh — Download a pre-trained lfmc weight file from Hugging Face
#
# Usage:
#   ./scripts/download-model.sh [MODEL_ID] [REVISION]
#
# Examples:
#   ./scripts/download-model.sh                          # default: lfm-tiny
#   ./scripts/download-model.sh soasme/lfmc-weights
#   ./scripts/download-model.sh soasme/lfmc-weights main lfm-small.bin
#
# The script saves the downloaded weight file to weights/<filename>.
# Requires: curl (or wget as fallback). Optional: sha256sum for verification.

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────
HF_REPO="${1:-soasme/lfmc-weights}"
REVISION="${2:-main}"
FILENAME="${3:-lfm-tiny.bin}"

HF_URL="https://huggingface.co/${HF_REPO}/resolve/${REVISION}/${FILENAME}"
DEST="weights/${FILENAME}"

# ── Helpers ─────────────────────────────────────────────────────────────────
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
info()  { printf '\033[36m==> %s\033[0m\n' "$*"; }

mkdir -p weights

# ── Check dependencies ───────────────────────────────────────────────────────
if command -v curl &>/dev/null; then
    DOWNLOADER=curl
elif command -v wget &>/dev/null; then
    DOWNLOADER=wget
else
    red "Error: neither curl nor wget found. Please install one."
    exit 1
fi

# ── Download ─────────────────────────────────────────────────────────────────
info "Repo:     ${HF_REPO} (revision: ${REVISION})"
info "File:     ${FILENAME}"
info "URL:      ${HF_URL}"
info "Dest:     ${DEST}"
echo

if [[ -f "${DEST}" ]]; then
    echo "File already exists at ${DEST}"
    read -r -p "Re-download? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || { echo "Skipped."; exit 0; }
fi

if [[ "${DOWNLOADER}" == "curl" ]]; then
    curl -L --progress-bar \
         -H "Accept: application/octet-stream" \
         "${HF_URL}" -o "${DEST}"
else
    wget --show-progress -q "${HF_URL}" -O "${DEST}"
fi

# ── Verify magic bytes ───────────────────────────────────────────────────────
# lfmc weight files start with magic 0x4C464D43 ("LFMC")
MAGIC=$(xxd -l 4 -p "${DEST}" 2>/dev/null || od -A n -N 4 -t x1 "${DEST}" | tr -d ' \n')
if [[ "${MAGIC,,}" == "4c464d43" ]]; then
    green "Magic bytes OK (LFMC format)"
else
    echo "Warning: unexpected magic bytes '${MAGIC}' — file may not be a valid lfmc weight file."
fi

# ── Optional SHA256 checksum ─────────────────────────────────────────────────
SHA_URL="https://huggingface.co/${HF_REPO}/resolve/${REVISION}/${FILENAME}.sha256"
SHA_TMP=$(mktemp)
if curl -sf "${SHA_URL}" -o "${SHA_TMP}" 2>/dev/null; then
    EXPECTED=$(cat "${SHA_TMP}")
    if command -v sha256sum &>/dev/null; then
        ACTUAL=$(sha256sum "${DEST}" | awk '{print $1}')
    elif command -v shasum &>/dev/null; then
        ACTUAL=$(shasum -a 256 "${DEST}" | awk '{print $1}')
    else
        ACTUAL=""
    fi
    if [[ -n "${ACTUAL}" ]]; then
        if [[ "${ACTUAL}" == "${EXPECTED}" ]]; then
            green "SHA256 OK"
        else
            red "SHA256 mismatch!"
            red "  expected: ${EXPECTED}"
            red "  actual:   ${ACTUAL}"
            exit 1
        fi
    fi
fi
rm -f "${SHA_TMP}"

echo
green "Downloaded → ${DEST}"
echo "Run inference with:"
echo "  ./lfmc infer --weights ${DEST} --prompt \"hello\""
