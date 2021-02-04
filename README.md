## Connect to the instance
```
export PROJECT_ID=persiangpt2
export INSTANCE_NAME=tpu-persian-3
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
export INSTANCE_NAME=tpu-persian-3
export TPU_ZONE=us-central1-f
export STORAGE_BUCKET=gs://persian-storage
export TPU_NAME=tpu-persian3
export MODEL_DIR=gs://persian-storage/experimentsSymbolic/
source ~/conda/bin/activate
```

## Generate the test data
```
python datasetTest.py
```

## Prepare the working environment
``` 
curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
chmod +x ~/miniconda.sh 
~/miniconda.sh -b -p ~/conda 
rm ~/miniconda.sh 
~/conda/bin/conda install -y python=3.6

source ~/conda/bin/activate

sudo apt-get install parallel
pip install tokenizers
sudo apt install git
sudo apt-get install tmux
sudo pip3 install -U tpunicorn

export PYTHONPATH="${PYTHONPATH}:/home/m5valipo/conda/bin/python"

pip install -r requirements.txt
pip install cloud-tpu-client

pip install tensorflow-serving-api
pip install cloud-tpu-client
pip install --upgrade google-api-python-client
pip install google-cloud-storage
pip install --upgrade oauth2client
``` 

## Prepare the data
```
bash prepare_data.sh ./datapath ./outputpath
bash prepare_data_pt.sh ./datapath ./outputpath
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
python train.py --config_file=configs/base.json --input_file="D:/Datasets/Symbolic Dataset/Processed/1VARPT/data_1024/*.tfrecord" --output_dir="D:/experiments/base/" --max_seq_length=1024 --train_batch_size=2 --learning_rate=1e-4 --num_train_steps=1000000 --num_warmup_steps=10000 --save_checkpoints_steps=1000 --iterations_per_loop=1000 --init_checkpoint="" --gradient_accumulation=20 --numberofPoints=30 --numberofVars=1 --model_type=PT
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
- [x] Generate dataset with the following parameters: Maximum NumberVariables: 5, Decimals: 5, numberofPoints: 30
- [ ] Just save the best models

# REFERENCES: 
- https://github.com/agermanidis/OpenGPT-2
- https://github.com/imcaspar/gpt2-ml
- https://huggingface.co/blog/how-to-train
- https://github.com/bhargaviparanjape/clickbait
- https://github.com/hpandana/gradient-accumulation-tf-estimator