#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --mem=1G

singularity exec $SIM_FOLDER/d3rlpy.sif pip install pandas scikit-learn

REWARDS=(1 2 3)
MODELS=('BCQ' 'DDPG' 'TD3' 'SAC' 'CQL' 'CRR')
for r in "${REWARDS[@]}"
do
  for m in "${MODELS[@]}"
  do
    sbatch -p scavenger-gpu -o "${DEEP_RL_LOG_FOLDER}/${m}_r${r}_SIM_${SIM_NUM}.out" -e "${DEEP_RL_LOG_FOLDER}/${m}_r${r}_SIM_${SIM_NUM}.err" --mem="$DEEP_RL_MEMORY" --export=ALL,MODEL_TYPE=$m,REWARD=$r $DEEP_RL_FOLDER/slurm_deep_rl_run.q
  done
done
