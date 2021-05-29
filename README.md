
This is the code for our proposed method, SymbolicGPT, and we tried to keep the implementation as simple and clean as possible to make sure it's understandable and easy to re-implement. 

# Abstract:
Symbolic regression is the task of identifying a mathematical expression that best fits a provided dataset of input and output values. Due to the richness of the space of mathematical expressions, symbolic regression is generally a challenging problem. While conventional approaches based on genetic evolution algorithms have been used for decades, deep learning-based methods are relatively new and an active area of research. In this work, we present a novel transformer-based language model for symbolic regression. This model exploits the strength and other possible flexibilities that have been provided by probabilistic language models like GPT. We show that our model is state of the art in terms of scalability and performance through comprehensive experiments.

# Dataset Generation

Skip this part, if you want to use the already generated datasets in this repository. Just make sure to extract the datasets, and change the configuration.

## How to generate the training data:
```bash
$ cd generator
$ for P in {1..2} ; do sleep 1;  echo $P; python dataset.py ; done
```

## Generate the test data:
```bash
$ cd generator
$ python datasetTest.py
```

# Train/Test the model

It's easy to train a new model and reproduce the results.

## Configure the parameters

Follow each dataset config file and change the parameters in the symbolicGPT.py script. 

## Run the script
```bash
python symbolicGPT.py
```

# TODO - Requirements:

# System Spec:
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- Single NVIDIA GeForce RTX 2080 11 GB
- 32.0 GB Ram

# Citation:
TODO: add after publication

# REFERENCES: 
- https://github.com/agermanidis/OpenGPT-2
- https://github.com/imcaspar/gpt2-ml
- https://huggingface.co/blog/how-to-train
- https://github.com/bhargaviparanjape/clickbait
- https://github.com/hpandana/gradient-accumulation-tf-estimator
- https://github.com/karpathy/minGPT
- https://github.com/charlesq34/pointnet
- https://github.com/volpato30/PointNovo
- https://github.com/brencej/ProGED

# License:
MIT