#!/bin/bash

cd ..

# custom config
DATA=/path/to/datasets
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

for SEED in 1 2 3
do
    if [ ${DATASET} == imagenet ]; then
        TESTMODEL=last_step
    else
        TESTMODEL=best_val
    fi
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.OURS.N_CTX ${NCTX} \
        TRAINER.OURS.CSC ${CSC} \
        TRAINER.OURS.CLASS_TOKEN_POSITION ${CTP} \
        TEST.FINAL_MODEL ${TESTMODEL} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done