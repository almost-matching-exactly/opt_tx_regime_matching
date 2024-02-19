#!/bin/bash
#
#SBATCH --get-user-env

source $CONDA_SOURCE
conda activate $CONDA_ENV

python $SIM_FOLDER/run_create_df.py $DATA_SAVE_FOLDER $SIM_NUM $N_ITERS $DGP_VERSION $N_PATIENTS $N_COVS $TIMSTEPS $TIMEDROPS $BINARY_DOSES $N_FOLDS $N_NEIGHBORS $N_BINS

sbatch -o "${DEEP_RL_LOG_FOLDER}/SIM_${SIM_NUM}_master.out" -e "${DEEP_RL_LOG_FOLDER}/SIM_${SIM_NUM}_master.err" $DEEP_RL_FOLDER/slurm_deep_rl.q

sh $R_FOLDER/run.sh

python $SIM_FOLDER/run_sim.py $DATA_SAVE_FOLDER $SIM_NUM $N_ITERS $BINARY_DOSES $N_FOLDS $N_NEIGHBORS $N_BINS
