#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

## bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500} ## training jobs with slurm

## just replacing config file path prefix
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

## the command line switch -m to allow modules to be located using the Python module namespace for execution as scripts
## `torch.distributed.launch` can be used to launch multiple processes per node for distributed training
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${PY_ARGS}
