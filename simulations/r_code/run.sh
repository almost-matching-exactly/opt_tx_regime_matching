#!/bin/bash

for (( iter=0; iter<$N_ITERS; iter++))
do
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-linear-r1
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-linear-r1
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-poly-r1
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-poly-r1
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-linear-r2
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-linear-r2
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-poly-r2
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-poly-r2
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-linear-r3
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-linear-r3
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-poly-r3
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter bowl-poly-r3
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter q-linear
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter q-linear
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter q-rpart
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter q-rpart
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter optclass-linear
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter optclass-linear
  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter optclass-rpart
  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter optclass-rpart
#  Rscript $R_FOLDER/DynTxRegime.R $DATA_SAVE_FOLDER $SIM_NUM $iter causaltree
#  python $R_FOLDER/calc_outcomes.py $DATA_SAVE_FOLDER $SIM_NUM $iter causaltree
  echo "R Iter $iter complete."
done