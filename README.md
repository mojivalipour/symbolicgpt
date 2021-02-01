## Connect to the instance
```
export PROJECT_ID=persiangpt2
export INSTANCE_NAME=tpu-persian
export TPU_ZONE=us-central1-f
gcloud compute ssh $INSTANCE_NAME --zone=$TPU_ZONE
```

```
git clone https://m5valipo:1ezHio5Rff6y-GET5drm@git.uwaterloo.ca/data-analytics-lab/symbolicgpt2.git
cd symbolicgpt2
```

```
pu list
cd symbolicgpt2
export INSTANCE_NAME=tpu-persian
export TPU_ZONE=us-central1-f
export STORAGE_BUCKET=gs://persian-storage
export TPU_NAME=tpu-persian2
export MODEL_DIR=gs://persian-storage/experimentsSymbolic/
source ~/conda/bin/activate
```

## How to generate the data
``` 
for P in {1..200} ; do sleep 1;  echo $P; python dataset.py ; done
```

# Train GPT2 Model using GPU
``` 
python train.py --config_file=configs/base.json --input_file="D:/Datasets/Symbolic Dataset/Processed/GPT2/data_1024/*.tfrecord" --output_dir="D:/experiments/base/" --max_seq_length=1024 --train_batch_size=2 --learning_rate=1e-4 --num_train_steps=1000000 --num_warmup_steps=10000 --save_checkpoints_steps=1000 --iterations_per_loop=1000 --init_checkpoint="" --gradient_accumulation=20
```

# Train PT Model using GPU
``` 
python train.py --config_file=configs/base.json --input_file="D:/Datasets/Symbolic Dataset/Processed/PT/data_1024/*.tfrecord" --output_dir="D:/experiments/base/" --max_seq_length=1024 --train_batch_size=2 --learning_rate=1e-4 --num_train_steps=1000000 --num_warmup_steps=10000 --save_checkpoints_steps=1000 --iterations_per_loop=1000 --init_checkpoint="" --gradient_accumulation=20 --numberofPoints=30 --numberofVars=5 --model_type='PT'
```

# TODO: 
- [x] Add the constant (Maysum)
- [x] Generate Data with multiple variables (Moji)
- [x] Implement GPT+PointNET (PT) (Moji)
- [x] Train the model with the new data (Moji)
- [x] MSE PLOT Code (Moji)
- [x] Test cases, limited number (1000)
- [x] MSE + Cumulative Implementation Plot (Moji)
- [ ] Find Interesting Showcases
- [ ] Compare with other methods (GPLearn: Python Package)
- [ ] Graph showing running time (CPU/GPU)
- [x] Maximum NumberVariables: 5
- [x] decimals: 4

# REFERENCES: 
- https://github.com/agermanidis/OpenGPT-2
- https://github.com/imcaspar/gpt2-ml
- https://huggingface.co/blog/how-to-train
- https://github.com/bhargaviparanjape/clickbait
- https://github.com/hpandana/gradient-accumulation-tf-estimator