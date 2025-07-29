#!/bin/bash

# === CONFIGURATION ===
REMOTE_USER="fleroux"
REMOTE_HOST="loom-elt2.lam.fr"
REMOTE_VENV="/net/SRVSTK12C/harmoni/fleroux/code/environments/env-francois"

# Local script to upload
LOCAL_SCRIPT="./compute_high_res_KL.py"

# Remote destination for the script
REMOTE_TMP_SCRIPT="/net/SRVSTK12C/harmoni/fleroux/code/tmp/compute_high_res_KL_remote.py"

# Unique tmux session name (timestamp to avoid collisions)
TMUX_SESSION="remote_py_$(date +%s)"

# === ENSURE REMOTE TMP DIRECTORY EXISTS ===
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p $(dirname "${REMOTE_TMP_SCRIPT}")"

# === COPY SCRIPT TO REMOTE MACHINE ===
echo "Uploading ${LOCAL_SCRIPT} to ${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}..."
scp "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

# === RUN SCRIPT IN TMUX AND KEEP SESSION OPEN AFTER ===
echo "Running script in tmux session '${TMUX_SESSION}'..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "tmux new-session -d -s ${TMUX_SESSION} \
  'source \"${REMOTE_VENV}/bin/activate\" && \
   python \"${REMOTE_TMP_SCRIPT}\" ; \
   echo \"Python script exited with status \$?\" ; \
   rm -f \"${REMOTE_TMP_SCRIPT}\" ; \
   exec bash'"

# === INFO ===
echo "Script is running in tmux session '${TMUX_SESSION}' on ${REMOTE_HOST}."
echo "You can attach to this session with:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  tmux attach -t ${TMUX_SESSION}"
echo ""
echo "Detach from tmux anytime with Ctrl+b then d."
echo "Exit the tmux session by typing 'exit' or Ctrl+d inside it."
