#!/usr/bin/env python
# coding: utf-8

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# load libraries
import os
import glob
import json
import pickle
import math
import random
import numpy as np
from tqdm import tqdm
from numpy import * # to override the math functions

import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import Dataset

from utils import set_seed
from matplotlib import pyplot as plt
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig
from utils import processDataFiles, CharDataset, tokenize_predict_and_evaluate, plot_and_save_results

# set the random seed
seed = 42
set_seed(seed)

# NOTE: make sure your data points are in the same range with trainRange

# config
numTests = 10 # number of times to generate candidates for one test equation
scratch=False # if you want to ignore the cache and start for scratch
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=[30,31] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=1 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 64 # spatial extent of the model for its context
testBlockSize = 400
batchSize = 64 # batch size of training data
const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
decimals = 8 # decimals of the points only if target is Skeleton
trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton
testRange = [[-5.0, 3.0],[-3.0, 5.0]]
useRange = True
dataDir = 'D:/Datasets/Symbolic Dataset/Datasets/FirstDataGenerator/' #'./datasets/'
dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize'.format(numVars, numPoints[0], numPoints[1], embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points'
dataTestFolder = '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
# EMB_CAT: Concat point embedding to GPT token+pos embedding
# EMB_SUM: Add point embedding to GPT tokens+pos embedding
# OUT_CAT: Concat the output of the self-attention and point embedding
# OUT_SUM: Add the output of the self-attention and point embedding
# EMB_CON: Conditional Embedding, add the point embedding as the first token
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
# NOT_VAR: Do nothing, will not pass any information from the number of variables in the equation to the GPT
# LEA_EMB: Learnable embedding for the variables, added to the pointNET embedding
# STR_VAR: Add the number of variables to the first token
addVars = True if variableEmbedding == 'STR_VAR' else False
maxNumFiles = 100 # maximum number of file to load in memory for training the neural network
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_SymbolicGPT_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                             'GPT_PT_{}_{}'.format(method, target), 
                                             'Padding',
                                             variableEmbedding)

def main(resultDict, modelKey):
    ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
    try: 
        os.mkdir(addr)
    except:
        print('Folder already exists!')

    # load the train dataset
    train_file = 'train_dataset_{}.pb'.format(fName)
    if os.path.isfile(train_file) and not scratch:
        # just load the train set
        with open(train_file, 'rb') as f:
            train_dataset,trainText,chars = pickle.load(f)
    else:
        # process training files from scratch
        path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
        files = glob.glob(path)[:maxNumFiles]
        text = processDataFiles(files)
        chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
        text = text.split('\n') # convert the raw text to a set of examples
        trainText = text[:-1] if len(text[-1]) == 0 else text
        random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
        train_dataset = CharDataset(trainText, blockSize, chars, numVars=numVars, 
                                    numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                                    const_range=const_range, xRange=trainRange, decimals=decimals) 
        with open(train_file, 'wb') as f:
            pickle.dump([train_dataset,trainText,chars], f)

    # print a random sample
    idx = np.random.randint(train_dataset.__len__())
    inputs, outputs, points, variables = train_dataset.__getitem__(idx)
    print('inputs:{}'.format(inputs))
    inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

    # load the val dataset
    path = '{}/{}/Val/*.json'.format(dataDir,dataFolder)
    files = glob.glob(path)
    textVal = processDataFiles([files[0]])
    textVal = textVal.split('\n') # convert the raw text to a set of examples
    val_dataset = CharDataset(textVal, blockSize, chars, numVars=numVars, 
                            numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                            const_range=const_range, xRange=trainRange, decimals=decimals)

    # print a random sample
    idx = np.random.randint(val_dataset.__len__())
    inputs, outputs, points, variables = val_dataset.__getitem__(idx)
    print(points.min(), points.max())
    inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

    # load the test data
    path = '{}/{}/Test/*.json'.format(dataDir,dataTestFolder)
    files = glob.glob(path)
    textTest = processDataFiles(files)
    textTest = textTest.split('\n') # convert the raw text to a set of examples
    # test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
    test_dataset = CharDataset(textTest, testBlockSize, chars, numVars=numVars, 
                            numYs=numYs, numPoints=numPoints, addVars=addVars,
                            const_range=const_range, xRange=trainRange, decimals=decimals)

    # print a random sample
    idx = np.random.randint(test_dataset.__len__())
    inputs, outputs, points, variables = test_dataset.__getitem__(idx)
    print(points.min(), points.max())
    inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

    # create the model
    pconf = PointNetConfig(embeddingSize=embeddingSize, 
                        numberofPoints=numPoints[1]-1, 
                        numberofVars=numVars, 
                        numberofYs=numYs,
                        method=method,
                        variableEmbedding=variableEmbedding)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=8, n_head=8, n_embd=embeddingSize, 
                    padding_idx=train_dataset.paddingID)
    model = GPT(mconf, pconf)
        
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=1, batch_size=batchSize, 
                        learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20, 
                        final_tokens=2*len(train_dataset)*blockSize,
                        num_workers=0, ckpt_path=ckptPath)
    trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss)

    # load the best model
    model.load_state_dict(torch.load(ckptPath))
    model = model.eval().to(trainer.device)
    print('{} model has been loaded!'.format(ckptPath))

    ## Test the model
    # alright, let's sample some character-level symbolic GPT 
    loader = torch.utils.data.DataLoader(
                                    test_dataset, 
                                    shuffle=False, 
                                    pin_memory=True,
                                    batch_size=1,
                                    num_workers=0)

    try:
        # for i, (inputs,outputs,points,variables) in tqdm(enumerate(loader), total=len(test_dataset)):
        #     tokenize_predict_and_evaluate(
        #                         i, inputs, points, outputs, variables, 
        #                         train_dataset, textTest, trainer, model, 
        #                         resultDict, numTests, variableEmbedding, 
        #                         blockSize, fName, modelKey=modelKey)
        from multiprocessing import Process
        processes = []
        for i, (inputs,outputs,points,variables) in tqdm(enumerate(loader), total=len(test_dataset)):
            proc = Process(target=tokenize_predict_and_evaluate(
                            i, inputs, points, outputs, variables, 
                            train_dataset, textTest, trainer, model, 
                            resultDict, numTests, variableEmbedding, 
                            blockSize, fName, modelKey=modelKey))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    plot_and_save_results(resultDict, fName, pconf, titleTemplate, 
                          textTest, modelKey=modelKey)

if __name__ == '__main__':
    modelKey='SymbolicGPT'
    resultDict = {}
    resultDict[fName] = {modelKey:{'err':[],'trg':[],'prd':[]}}
    main(resultDict, modelKey)

    