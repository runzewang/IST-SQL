#! /bin/bash
devices=$1
model_type=$2
LOGDIR=$3
sh run_cosql_${model_type}.sh $devices ${LOGDIR} &