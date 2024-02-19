#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --mem=1G

set -o allexport && source sim_vars.env && set +o allexport

s=0
for d in "${ALL_DGPS[@]}"
do
  for c in "${ALL_COVS[@]}"
  do
    for b in "${ALL_BINARY_DOSES[@]}"
    do
      for t in "${ALL_NUM_T[@]}"
      do
        for dr in "${ALL_NUM_T_DROP[@]}"
        do
          sbatch -o "${LOG_FOLDER}/SIM_${s}.out" -e "${LOG_FOLDER}/SIM_${s}.err" --mem="$MEMORY" --export=ALL,SIM_NUM=$s,DGP_VERSION=$d,N_COVS=$c,TIMSTEPS=$t,TIMEDROPS=$dr,BINARY_DOSES=$b slurm_sim_run.q
          s=$((s+1))
        done
      done
    done
  done
done
