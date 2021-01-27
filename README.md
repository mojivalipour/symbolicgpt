## Connect to the instance
```
export PROJECT_ID=persiangpt2
export INSTANCE_NAME=tpu-persian
export TPU_ZONE=us-central1-f
gcloud compute ssh $INSTANCE_NAME --zone=$TPU_ZONE
```

git clone https://m5valipo:1ezHio5Rff6y-GET5drm@git.uwaterloo.ca/data-analytics-lab/symbolicgpt2.git
cd symbolicgpt2

pu list
cd symbolicgpt2
export INSTANCE_NAME=tpu-persian
export TPU_ZONE=us-central1-f
export STORAGE_BUCKET=gs://persian-storage
export TPU_NAME=tpu-persian2
export MODEL_DIR=gs://persian-storage/experimentsSymbolic/
source ~/conda/bin/activate

# TODO: 
- [x] Add the constant (Maysum)
- [x] Generate Data with multiple variables (Moji)
- [ ] Train the model with the new data (Moji)
- [ ] MSE PLOT Code (Maysum)
- [ ] Test cases, limited number (1000)
- [ ] MSE + Cumulative Implementation Plot
- [ ] Find Interesting Showcases
- [ ] Compare with other methods (GPLearn: Python Package)
- [ ] Graph showing running time (CPU/GPU)
- [x] Maximum NumberVariables: 5
- [x] decimals: 4