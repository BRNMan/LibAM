#!/bin/bash
# ida_process.sh — Upload binaries, run IDA + feature extraction on remote, download results.
#
# Usage:
#   ./ida_process.sh <local_target_folder> <local_candidate_folder> <password_file> <remote_host>
#
# Example:
#   ./ida_process.sh ./dataset2/1_binary/target ./dataset2/1_binary/candidate ./pass.txt gpu-server.cc.gatech.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 <local_target_folder> <local_candidate_folder> <password_file> <remote_host>

Arguments:
  local_target_folder     Path to local folder containing target binaries
  local_candidate_folder  Path to local folder containing candidate binaries
  password_file           File containing the SSH/SFTP password (single line)
  remote_host             Hostname or IP of the remote server

Environment variables:
  START_STEP=N  Skip to step N (1=upload, 2=upload candidate, 3=process, 4=download). Default: 1.
  DEBUG=1       Print verbose SSH output on connection failure.
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
[ "$#" -lt 4 ] && usage

LOCAL_TARGET="${1%/}"
LOCAL_CANDIDATE="${2%/}"
PASSWORD_FILE="$3"
REMOTE_HOST="$4"

REMOTE_USER="mbraun39"
REMOTE_HOME="/nethome/mbraun39"

# Strip user@ prefix from REMOTE_HOST if the caller passed user@host
if [[ "$REMOTE_HOST" == *@* ]]; then
    REMOTE_HOST="${REMOTE_HOST#*@}"
fi

[ -d "$LOCAL_TARGET" ]    || die "Local target folder not found: $LOCAL_TARGET"
[ -d "$LOCAL_CANDIDATE" ] || die "Local candidate folder not found: $LOCAL_CANDIDATE"
[ -f "$PASSWORD_FILE" ]   || die "Password file not found: $PASSWORD_FILE"

# Build a temporary askpass helper that prints the password.
# SSH_ASKPASS_REQUIRE=force tells OpenSSH (>= 8.4) to always use it,
# with no need for sshpass or any other extra tool.
ASKPASS=$(mktemp /tmp/askpass_XXXXXX)
chmod 700 "$ASKPASS"
cat > "$ASKPASS" <<ASKPASS_EOF
#!/bin/bash
head -1 "$PASSWORD_FILE" | tr -d '\r'
ASKPASS_EOF
export SSH_ASKPASS="$ASKPASS"
export SSH_ASKPASS_REQUIRE=force
trap 'rm -f "$ASKPASS"' EXIT

TARGET_NAME=$(basename "$LOCAL_TARGET")
CANDIDATE_NAME=$(basename "$LOCAL_CANDIDATE")
REMOTE_TARGET="$REMOTE_HOME/$TARGET_NAME"
REMOTE_CANDIDATE="$REMOTE_HOME/$CANDIDATE_NAME"

SSH_OPTS="-XC -o StrictHostKeyChecking=no"
SFTP_OPTS="-C -o StrictHostKeyChecking=no"  # sftp does not support -X

# Which step to start from (set START_STEP=3 to skip uploads, etc.)
START_STEP=${START_STEP:-1}

log "Remote target path:    $REMOTE_TARGET"
log "Remote candidate path: $REMOTE_CANDIDATE"



# ---------------------------------------------------------------------------
# Preflight: verify credentials before doing any real work.
# If this fails, re-run with DEBUG=1 to see full SSH verbose output.
# ---------------------------------------------------------------------------
log "=== [0/4] Testing SSH connection ==="
if ! ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" true 2>/dev/null; then
    if [ "${DEBUG:-0}" = "1" ]; then
        log "--- Verbose SSH output (DEBUG=1) ---"
        ssh -v $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" true || true
    fi
    die "SSH connection failed. Re-run with DEBUG=1 to see details:\n  DEBUG=1 $0 $*"
fi
log "Connection OK."

# ---------------------------------------------------------------------------
# Helper: run an sftp batch non-interactively
# ---------------------------------------------------------------------------
sftp_run() {
    sftp $SFTP_OPTS "$REMOTE_USER@$REMOTE_HOST"
}

# ---------------------------------------------------------------------------
# 1. Upload target folder
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
    log "=== [1/4] Uploading target folder to remote server ==="
    sftp_run <<SFTP
put -r "$LOCAL_TARGET" .
quit
SFTP
else
    log "=== [1/4] Skipping upload of target folder (START_STEP=$START_STEP) ==="
fi

# ---------------------------------------------------------------------------
# 2. Upload candidate folder
# ---------------------------------------------------------------------------
if [ "$START_STEP" -le 2 ]; then
    log "=== [2/4] Uploading candidate folder to remote server ==="
    sftp_run <<SFTP
put -r "$LOCAL_CANDIDATE" .
quit
SFTP
else
    log "=== [2/4] Skipping upload of candidate folder (START_STEP=$START_STEP) ==="
fi

# ---------------------------------------------------------------------------
# 3. Remote processing over a single SSH session
#    Note: IDA and feature extraction can each take tens of minutes.
# ---------------------------------------------------------------------------
log "=== [3/4] Running remote IDA processing and feature extraction ==="

ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" bash -s <<REMOTE
set -e

# ---- Source IDA environment into this bash session ----
# idapro.csh is a tcsh script; run it in tcsh, capture the resulting env,
# then export every variable into bash so all subsequent commands see IDA.
echo "[remote] Sourcing IDA environment..."
eval "\$(tcsh -c 'source /tools/software/hex-rays/idapro.csh >& /dev/null && env' 2>/dev/null \
    | grep -E '^(PATH|IDADIR|IDA[A-Z_]*|LD_LIBRARY_PATH)=' \
    | sed 's/^/export /')"

# ---- IDA: generate .i64 files ----
echo "[remote] Generating .i64 files from target binaries..."
find "$REMOTE_TARGET"/*/* -exec idat -A -S"exit.py" {} \;

echo "[remote] Generating .i64 files from candidate binaries..."
find "$REMOTE_CANDIDATE"/*/* -exec idat -A -S"exit.py" {} \;

# ---- Feature extraction ----
echo "[remote] Extracting FCG features for target..."
python3 ~/run_get_features.py "$REMOTE_TARGET" fcg

echo "[remote] Extracting function features for target..."
python3 ~/run_get_features.py "$REMOTE_TARGET" feature

echo "[remote] Extracting FCG features for candidate..."
python3 ~/run_get_features.py "$REMOTE_CANDIDATE" fcg

echo "[remote] Extracting function features for candidate..."
python3 ~/run_get_features.py "$REMOTE_CANDIDATE" feature

# ---- Organise target output: 2_target/{fcg,feature} ----
echo "[remote] Organising target output files..."
mkdir -p "$REMOTE_TARGET/2_target/fcg"
mkdir -p "$REMOTE_TARGET/2_target/feature"
find "$REMOTE_TARGET" -maxdepth 3 -name "*_fcg.pkl" \
    ! -path "*/2_target/*" \
    -exec mv {} "$REMOTE_TARGET/2_target/fcg/" \;
find "$REMOTE_TARGET" -maxdepth 3 -name "*.json" \
    ! -path "*/2_target/*" \
    -exec mv {} "$REMOTE_TARGET/2_target/feature/" \;

# ---- Organise candidate output: 3_candidate/{fcg,feature} ----
echo "[remote] Organising candidate output files..."
mkdir -p "$REMOTE_CANDIDATE/3_candidate/fcg"
mkdir -p "$REMOTE_CANDIDATE/3_candidate/feature"
find "$REMOTE_CANDIDATE" -maxdepth 3 -name "*_fcg.pkl" \
    ! -path "*/3_candidate/*" \
    -exec mv {} "$REMOTE_CANDIDATE/3_candidate/fcg/" \;
find "$REMOTE_CANDIDATE" -maxdepth 3 -name "*.json" \
    ! -path "*/3_candidate/*" \
    -exec mv {} "$REMOTE_CANDIDATE/3_candidate/feature/" \;

echo "[remote] Processing complete."
REMOTE

# ---------------------------------------------------------------------------
# 4. Download results
# ---------------------------------------------------------------------------
log "=== [4/4] Downloading processed results ==="

log "Downloading 2_target..."
sftp_run <<SFTP
get -r "$REMOTE_TARGET/2_target" "$LOCAL_TARGET/"
quit
SFTP

log "Downloading 3_candidate..."
sftp_run <<SFTP
get -r "$REMOTE_CANDIDATE/3_candidate" "$LOCAL_CANDIDATE/"
quit
SFTP

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "=== Complete! Results saved to:"
log "    $LOCAL_TARGET/2_target/"
log "    $LOCAL_CANDIDATE/3_candidate/"
