#!/usr/bin/env bash
# Fetch the Cora citation edge list (real directed graph, 2708 nodes / 5429 edges).
# Source: LINQS. cora.cites is `cited_id citing_id` per line. Data lands in
# graphops/data/ which is gitignored.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
URL="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

mkdir -p "$DEST"
if [ -f "$DEST/cora/cora.cites" ]; then
  echo "have cora.cites"
  exit 0
fi
echo "fetching cora.tgz"
curl -sSL --fail -o "$DEST/cora.tgz" "$URL"
tar xzf "$DEST/cora.tgz" -C "$DEST"
rm -f "$DEST/cora.tgz"
echo "done -> $DEST/cora"
