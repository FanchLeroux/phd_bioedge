#!/bin/bash

# === CONFIGURATION ===
REMOTE_USER="fleroux"
REMOTE_HOST="loom-elt2.lam.fr"
REMOTE_VENV="/net/SRVSTK12C/harmoni/fleroux/code/environments/env-francois"

# Local Python script to upload
LOCAL_SCRIPT="./compute_high_res_KL.py"

# Remote temporary directory and script location
REMOTE_TMP_DIR="/net/SRVSTK12C/harmoni/fleroux/code/tmp"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_DIR}/compute_high_res_KL_remote.py"

# Remote output file path (must match the output of your Python script)
REMOTE_OUTPUT_FILE="${REMOTE_TMP_DIR}/compute_high_res_KL_output.npy"

# File that signals the end of the script (created by tmux command)
REMOTE_DONE_FLAG="${REMOTE_TMP_DIR}/done.flag"

# Local path where output will be saved
LOCAL_OUTPUT_DIR="./results"
LOCAL_OUTPUT_FILE="${LOCAL_OUTPUT_DIR}/$(basename "${REMOTE_OUTPUT_FILE}")"

# Unique tmux session name
TMUX_SESSION="remote_py_$(date +%s)"

# === PREPARE REMOTE DIRECTORY ===
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p \"${REMOTE_TMP_DIR}\""

# === COPY PYTHON SCRIPT TO REMOTE ===
echo "Uploading ${LOCAL_SCRIPT} to ${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}..."
scp "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

# === RUN PYTHON SCRIPT IN A PERSISTENT TMUX SESSION ===
echo "Launching Python script in tmux session '${TMUX_SESSION}'..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "tmux new-session -d -s ${TMUX_SESSION} \
  'source \"${REMOTE_VENV}/bin/activate\" && \
   python \"${REMOTE_TMP_SCRIPT}\" && \
   touch \"${REMOTE_DONE_FLAG}\" ; \
   echo \"Python script finished.\" ; \
   exec bash'"

# === WAIT FOR SIGNAL FILE ===
echo "Waiting for remote script to complete (watching for done.flag)..."
until ssh "${REMOTE_USER}@${REMOTE_HOST}" "[ -f \"${REMOTE_DONE_FLAG}\" ]"; do
    sleep 5
done

echo "Remote script completed. Downloading output file..."

# === DOWNLOAD THE RESULT ===
mkdir -p "${LOCAL_OUTPUT_DIR}"
scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUT_FILE}" "${LOCAL_OUTPUT_FILE}"

# === CLEANUP DONE FLAG ===
ssh "${REMOTE_USER}@${REMOTE_HOST}" "rm -f \"${REMOTE_DONE_FLAG}\""

echo "Output file downloaded to: ${LOCAL_OUTPUT_FILE}"
echo ""
echo "The tmux session '${TMUX_SESSION}' is still running on the remote machine."
echo ""
echo "To access the tmux session:"
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "   tmux attach -t ${TMUX_SESSION}"
echo ""
echo "To detach from tmux: Press Ctrl+b, then d"
echo "To exit the tmux session completely: Type 'exit' inside tmux"

echo "Done."
