#!/usr/bin/env bash

export PYTHONPATH=/home/m5valipo/symbolicGPT2

learning_rate=1e-4
init_checkpoint=""
max_seq_length=1024
save_checkpoint_steps=1000
gradient_accumulation=5 #8 #16
numberofPoints=30 
numberofVars=1
model_type="PT" # GPT2

# You can customize the training here
# mega, medium, or base
model_size="base"
OUTPUT_DIR="gs://persian-storage/expSymbolic/Mesh_Simple_GPT2_1024_Sorted_${model_type}/${model_size}/" # put your output directory here
input_file="gs://persian-storage/symbolic/meshGPT1024SortedPT/*.tfrecord" # put your input files here, it can also be something like "*.tfrecord"

if [ ${model_size} == "base" ]; then
    num_tpu_cores=8 #32
    batch_size_per_core=13 #16
elif [ ${model_size} == "large" ]; then
    num_tpu_cores=8 #128
    batch_size_per_core=4 #16 #4
elif [ ${model_size} == "mega" ]; then
    num_tpu_cores=256
    batch_size_per_core=2
fi

# there are 800000 -> 20k * 1024 examples so this translates to 20 epochs. seems ok and i can run for more if needed
num_train_steps=1000000 # approximately 20 epochs -> 53K * 20 = 1060000

# Make sure batch size scales.
let batch_size="$batch_size_per_core * $num_tpu_cores"
trap "exit" INT
while true
do
    timeout --foreground --signal=SIGKILL 11h python train.py \
    --config_file=configs/${model_size}.json \
    --input_file=${input_file} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${max_seq_length} \
    --train_batch_size=${batch_size} \
    --learning_rate=${learning_rate} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=10000 \
    --save_checkpoints_steps=${save_checkpoint_steps} \
    --iterations_per_loop=${save_checkpoint_steps} \
    --use_tpu=True \    
    --tpu_name=$TPU_NAME\    
    --num_tpu_cores=$num_tpu_cores \
    --init_checkpoint=${init_checkpoint}\
    --gradient_accumulation=${gradient_accumulation}\    
    --numberofPoints=${numberofPoints}\
    --numberofVars=${numberofVars}\
    --model_type=${model_type}
    echo restarting   
    sleep 30
done