#!/bin/bash
# Auto-extract Pod 3 phase-A bundle and regenerate paper figures + tables.
#
# Polls bundles/pod3_final_bundle.tar.gz (created by pull_pod3_final.sh once
# Pod 3's phase-A watcher fires).  When it appears, extracts to
# extracted/pod3/ and re-runs make_paper_figures.py + make_paper_tables.py
# so MemNO + F-FNO baselines join the heatmap, per-regime, and compute-cost
# tables.  Then runs post_hoc_analyses.py --aggregate to refresh F09-F12 if
# any per-cell post-hoc data has been written by other workers.

set -e
cd "/a/dde research/dde-fno"

BUNDLE=bundles/pod3_final_bundle.tar.gz
DEST=extracted/pod3

echo "[$(date +%H:%M:%S)] auto_extract: waiting for $BUNDLE..."
while [ ! -f "$BUNDLE" ]; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] auto_extract: $BUNDLE found ($(du -h $BUNDLE | cut -f1))"

# Wait for size to stabilize (in case scp still in progress).
prev_size=0
while true; do
  cur=$(stat -c %s "$BUNDLE")
  if [ "$cur" = "$prev_size" ] && [ "$cur" -gt 0 ]; then
    break
  fi
  prev_size="$cur"
  sleep 5
done
echo "[$(date +%H:%M:%S)] auto_extract: size stable at $prev_size bytes; extracting..."

mkdir -p "$DEST"
tar xzf "$BUNDLE" -C "$DEST" 2>&1 | tail -5

n_results=$(find "$DEST" -name test_results.json 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] auto_extract: $n_results test_results.json files extracted"

echo "[$(date +%H:%M:%S)] auto_extract: regenerating figures..."
python scripts/make_paper_figures.py 2>&1 | tail -15

echo "[$(date +%H:%M:%S)] auto_extract: regenerating tables..."
python scripts/make_paper_tables.py 2>&1 | tail -10

echo "[$(date +%H:%M:%S)] auto_extract: refreshing post-hoc aggregate..."
python scripts/post_hoc_analyses.py --aggregate \
  --layer_root extracted/pod3/outputs/final_baselines 2>&1 | tail -10 || true

echo "[$(date +%H:%M:%S)] auto_extract: DONE"
