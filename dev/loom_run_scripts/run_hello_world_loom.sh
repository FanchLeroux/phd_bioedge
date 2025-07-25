# ssh loom python /net/SRVSTK12C/harmoni/fleroux/test.py
# ssh loom bash -c /net/SRVSTK12C/harmoni/fleroux/run_test2.sh

ssh "loom" "bash -c 'source /net/SRVSTK12C/harmoni/fleroux/code/environments/env-francois/bin/activate && 
bash /net/SRVSTK12C/harmoni/fleroux/run_test2.sh'"
