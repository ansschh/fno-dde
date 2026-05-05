#!/bin/bash
# Pull post-sweep artifact tarballs from every RunPod cluster pod into local
# A:/dde research/dde-fno/extracted/<pod_label>/ for offline figure regeneration.
#
# Prerequisite on each pod: scripts/_post_sweep_batch.sh has finished and
# /workspace/post_sweep_artifacts_<hostname>.tar.gz exists.
#
# Usage:
#   bash scripts/_pull_post_sweep_artifacts.sh
#
# Skips pods whose tarball is missing. Verifies tar after pull, untars into
# extracted/<pod_label>/ so figure scripts can pick up the new cells.

set -uo pipefail
LOCAL_ROOT="A:/dde research/dde-fno"
EXTRACT="$LOCAL_ROOT/extracted"
mkdir -p "$EXTRACT"

# Pod label -> SSH spec (label is the local extract dir name).
# Direct-SSH pods take 2 args (port + host); proxy pods take 1 arg.
declare -A POD_DIRECT=(
    [large_scale]="43676 root@185.216.23.244"
    [wall_clock]="10808 root@209.170.80.132"
    [pod1]="22021 root@194.68.245.85"
    [pod2]="22174 root@194.68.245.54"
    [pod4]="27692 root@64.247.206.120"
    [pod5]="22037 root@194.68.245.21"
    [pod6]="22141 root@194.68.245.111"
    [pod_a]="17683 root@202.181.159.213"
)
declare -A POD_PROXY=(
    [pod_idle3]="0ka02fb815q6gm-64412158@ssh.runpod.io"
    [pod_b]="4cayh3fj66vfju-64412156@ssh.runpod.io"
    [pod_c]="krk5mgfmkptt9p-64412159@ssh.runpod.io"
)

SSHOPTS="-o ConnectTimeout=20 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519"

pull_one() {
    local label="$1"; local kind="$2"; local args="$3"
    local local_tarball="$EXTRACT/post_sweep_${label}.tar.gz"
    local local_dir="$EXTRACT/${label}"
    echo ""
    echo "=== [$label] kind=$kind ==="

    # Discover hostname for tarball naming. We need to ssh once to get $(hostname -s).
    local hostname_short=""
    if [ "$kind" = "direct" ]; then
        local port=$(echo "$args" | awk '{print $1}')
        local host=$(echo "$args" | awk '{print $2}')
        hostname_short=$(ssh $SSHOPTS -p "$port" "$host" 'hostname -s' 2>/dev/null || true)
    else
        # Proxy SSH needs PTY; pipe via stdin.
        hostname_short=$(echo 'hostname -s; exit' | timeout 30 ssh -tt $SSHOPTS "$args" 2>&1 \
            | grep -v "^\*\*\|RUNPOD\|----\|^---\|^_____\|^|\| | \||  __\||\\\\\|^For\|^https\|^Enjoy\|^For \|^[?]2004\|^Connection\|^$\|^ *$\|^ *_\|exit$" \
            | tail -1 | tr -d '\r')
    fi
    if [ -z "$hostname_short" ]; then
        echo "[$label]   SKIP: could not get remote hostname"
        return 1
    fi
    local remote_tarball="/workspace/post_sweep_artifacts_${hostname_short}.tar.gz"
    echo "[$label]   remote hostname=$hostname_short  tarball=$remote_tarball"

    # SCP. Direct SCP works; proxy SSH does NOT support SCP / SFTP — need a
    # different path for proxy pods (would have to base64-pipe the file).
    # For now, attempt SCP and warn on failure; user can manually pull
    # proxy-pod tarballs via the runpod web UI's "Download" tab.
    if [ "$kind" = "direct" ]; then
        local port=$(echo "$args" | awk '{print $1}')
        local host=$(echo "$args" | awk '{print $2}')
        echo "[$label]   scp..."
        scp $SSHOPTS -P "$port" -C "$host:$remote_tarball" "$local_tarball" \
            && echo "[$label]   scp OK  $(du -h "$local_tarball" | cut -f1)" \
            || { echo "[$label]   scp FAILED"; return 2; }
    else
        echo "[$label]   PROXY pod — SCP not supported. Use RunPod web UI 'Files'"
        echo "[$label]   tab to download $remote_tarball, save to $local_tarball"
        return 3
    fi

    # Verify + extract.
    if ! tar -tzf "$local_tarball" > /dev/null 2>&1; then
        echo "[$label]   tarball verify FAILED"; return 4
    fi
    mkdir -p "$local_dir"
    tar -xzf "$local_tarball" -C "$local_dir/" \
        && echo "[$label]   extracted to $local_dir/  ($(du -sh "$local_dir" | cut -f1))" \
        || echo "[$label]   tar -x FAILED"
}

echo "=== pull_post_sweep_artifacts START $(date) ==="
for label in "${!POD_DIRECT[@]}"; do
    pull_one "$label" direct "${POD_DIRECT[$label]}" || true
done
for label in "${!POD_PROXY[@]}"; do
    pull_one "$label" proxy "${POD_PROXY[$label]}" || true
done
echo "=== pull_post_sweep_artifacts DONE $(date) ==="
echo ""
echo "Local artifact roots:"
ls -d "$EXTRACT"/*/ 2>/dev/null
echo ""
echo "Re-render figures:"
echo "  cd 'A:/dde research/dde-fno'"
echo "  python scripts/make_phase2_figures.py"
echo "  python scripts/make_paper_figures.py"
echo "  python scripts/make_paper_tables.py"
