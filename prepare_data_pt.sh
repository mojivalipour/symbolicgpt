#!/usr/bin/env bash

NUM_FOLDS=1024
MAX_SEQ_LENGTH=1024
MAX_NUM_POINTS=30
MODEL_TYPE='PT'
FN=${1}
OUT_BUCKET=${2} # add gs:// for cloud 

rm -rf logs_${MAX_SEQ_LENGTH}
mkdir logs_${MAX_SEQ_LENGTH}
if [ "${2}" == "." ];then
    mkdir data_${MAX_SEQ_LENGTH}
fi
parallel -j $(nproc --all) --will-cite "python3 prepare_data.py -fold {1} -num_folds ${NUM_FOLDS} -modelType ${MODEL_TYPE} -max_num_points ${MAX_NUM_POINTS} -base_fn ${OUT_BUCKET}/data_${MAX_SEQ_LENGTH}/ -input_fn ${FN} -max_seq_length ${MAX_SEQ_LENGTH} > logs_${MAX_SEQ_LENGTH}/log{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))