#!/bin/bash

# === CONFIGURATION ===
REMOTE_USER="fleroux"
REMOTE_HOST="loom-elt2.lam.fr"
REMOTE_VENV="/net/SRVSTK12C/harmoni/fleroux/code/environments/env-francois"

# Local script to upload
LOCAL_SCRIPT="./hello_world.py"

# Remote destination for the script
REMOTE_TMP_SCRIPT="/net/SRVSTK12C/harmoni/fleroux/code/tmp/test_remote.py"

# === COPY SCRIPT TO REMOTE MACHINE ===
echo "Uploading ${LOCAL_SCRIPT} to ${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}..."
scp "${LOCAL_SCRIPT}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"

# === RUN SCRIPT ON REMOTE INSIDE VENV ===
echo "Running script on remote host..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "bash -c 'source \"${REMOTE_VENV}/bin/activate\" && python \"${REMOTE_TMP_SCRIPT}\"'"

# === CLEANUP  ===
echo "Cleaning up remote script..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "rm -f \"${REMOTE_TMP_SCRIPT}\""
echo "done."