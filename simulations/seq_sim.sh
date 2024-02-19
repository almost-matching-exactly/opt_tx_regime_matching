#!/bin/bash

set -o allexport && source sim_vars.env && set +o allexport

source $CONDA_SOURCE
conda activate $CONDA_ENV

s=0
for DGP_VERSION in "${ALL_DGPS[@]}"
do
  for N_COVS in "${ALL_COVS[@]}"
  do
    for BINARY_DOSES in "${ALL_BINARY_DOSES[@]}"
    do
      for TIMSTEPS in "${ALL_NUM_T[@]}"
      do
        for TIMEDROPS in "${ALL_NUM_T_DROP[@]}"
        do
          export SIM_NUM=$s
          python $SIM_FOLDER/run_create_df.py $DATA_SAVE_FOLDER $SIM_NUM $N_ITERS $DGP_VERSION $N_PATIENTS $N_COVS $TIMSTEPS $TIMEDROPS $BINARY_DOSES $N_FOLDS $N_NEIGHBORS $N_BINS
          sh $R_FOLDER/run.sh
          python $SIM_FOLDER/run_sim.py $DATA_SAVE_FOLDER $SIM_NUM $N_ITERS $BINARY_DOSES $N_FOLDS $N_NEIGHBORS $N_BINS
          s=$((s+1))
        done
      done
    done
  done
done
