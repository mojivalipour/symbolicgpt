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

from utils import set_seed, sample
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig
from utils import processDataFiles, CharDataset, relativeErr, mse, sqrt, divide, lossFunc

# set the random seed
seed = 42
set_seed(seed)

# config
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=250 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=9 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 400 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
trainRange = [-3.0,3.0] 
testRange = [[-5.0, 3.0],[-3.0, 5.0]]
useRange = True
decimals = 4
const_range = [-2.1, 2.1]
dataDir = 'D:/Datasets/Symbolic Dataset/Datasets/FirstDataGenerator/' #'./datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1-9Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_20-250'
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
maxNumFiles = 30 # maximum number of file to load in memory for training the neural network
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_SymbolicGPT_{}_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                             'GPT_PT_{}_{}'.format(method, target), 
                                             'Padding',
                                             blockSize,
                                             variableEmbedding)
ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
try: 
    os.mkdir(addr)
except:
    print('Folder already exists!')

# load the train dataset
train_file = 'train_dataset.pb'
if os.path.isfile(train_file):
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
                    numYs=numYs, numPoints=numPoints, target=target, addVars=addVars) 
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
                numYs=numYs, numPoints=numPoints, target=target, addVars=addVars)

# print a random sample
idx = np.random.randint(val_dataset.__len__())
inputs, outputs, points, variables = val_dataset.__getitem__(idx)
print(points.min(), points.max())
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

# load the test data
path = '{}/{}/Test/*.json'.format(dataDir,dataFolder)
files = glob.glob(path)
textTest = processDataFiles(files)
textTest = textTest.split('\n') # convert the raw text to a set of examples
# test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
test_dataset = CharDataset(textTest, blockSize, chars, numVars=numVars, 
                numYs=numYs, numPoints=numPoints, addVars=addVars)

# print a random sample
idx = np.random.randint(test_dataset.__len__())
inputs, outputs, points, variables = test_dataset.__getitem__(idx)
print(points.min(), points.max())
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

# create the model
pconf = PointNetConfig(embeddingSize=embeddingSize, 
                       numberofPoints=numPoints, 
                       numberofVars=numVars, 
                       numberofYs=numYs,
                       method=method,
                       variableEmbedding=variableEmbedding)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize, 
                  padding_idx=train_dataset.paddingID)
model = GPT(mconf, pconf)
    
# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize, 
                      learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2*len(train_dataset)*blockSize,
                      num_workers=0, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss)

# benchmarks
import csv

# rng = np.random.RandomState(seed)
# benchmarkPath = './benchmark/dsr-benchmark-data/'
# dataPoints = glob.glob(benchmarkPath+'*.csv')
# NGUYEN2Eq = {
#            '1':'x1**3+x1**2+x1',
#            '2':'x1**4+x1**3+x1**2+x1',
#            '3':'x1**5+x1**4+x1**3+x1**2+x1',
#            '4':'x1**6+x1**5+x1**4+x1**3+x1**2+x1',
#            '5':'sin(x1**2)*cos(x1)-1',
#            '6':'sin(x1)+sin(x1+x1**2)',
#            '7':'log(x1+1)+log(x1**2+1)',
#            '8':'sqrt(x1)',
#            '9':'sin(x1)+sin(x2**2)',
#            '10':'2*sin(x1)*cos(x2)',
#            '11':'x1**x2',
#            '12':'x1**4-x1**3+x2**2/2-x2',}

# NGUYEN2EqTemplates = {
#             '1':'C*x1**3+C*x1**2+C*x1+C',
#             '2':'C*x1**4+C*x1**3+C*x1**2+C*x1+C',
#             '3':'C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
#             '4':'C*x1**6+C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
#             '5':'C*sin(C*x1**2)*cos(C*x1+C)+C',
#             '6':'C*sin(C*x1+C)+C*sin(C*x1+C*x1**2)+C',
#             '7':'C*log(C*x1+C)+C*log(C*x1**2+C)+C',
#             '8':'C*sqrt(C*x1+C)+C',
#             '9':'C*sin(C*x1+C)+C*sin(C*x2**2+C)+C',
#             '10':'C*sin(C*x1+C)*cos(C*x2+C)+C',
#             '11':'C*x1**x2+C',
#             '12':'C*x1**4+C*x1**3+C*x2**2+C*x2+C',}

# NGUYENNumVars = {
#            '1':1,
#            '2':1,
#            '3':1,
#            '4':1,
#            '5':1,
#            '6':1,
#            '7':1,
#            '8':1,
#            '9':2,
#            '10':2,
#            '11':2,
#            '12':2,}

# bestEquations = {
#    '1':'C',
#    '2':'C',
#    '3':'C',
#    '4':'C',
#    '5':'C',
#    '6':'C',
#    '7':'C',
#    '8':'C',
#    '9':'C',
#    '10':'C',
#    '11':'C',
#    '12':'C',
# }

# bestTemplates = {
#    '1':'C',
#    '2':'C',
#    '3':'C',
#    '4':'C',
#    '5':'C',
#    '6':'C',
#    '7':'C',
#    '8':'C',
#    '9':'C',
#    '10':'C',
#    '11':'C',
#    '12':'C',
# }

# bestErr = {
#    '1':1000,
#    '2':1000,
#    '3':1000,
#    '4':1000,
#    '5':1000,
#    '6':1000,
#    '7':1000,
#    '8':1000,
#    '9':1000,
#    '10':1000,
#    '11':1000,
#    '12':1000,
# }

templatesEQs = {
        1: [
            # NGUYEN
            'C*x1**3+C*x1**2+C*x1+C', 
            'C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            'C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            'C*x1**6+C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            'C*sin(C*x1**2)*cos(C*x1+C)+C',
            'C*sin(C*x1+C)+C*sin(C*x1+C*x1**2)+C',
            'C*log(C*x1+C)+C*log(C*x1**2+C)+C',
            'C*sqrt(C*x1+C)+C',
            ],
        2: [
            # NGUYEN
            'C*sin(C*x1+C)+C*sin(C*x2**2+C)+C',
            'C*sin(C*x1+C)*cos(C*x2+C)+C',
            'C*x1**x2+C',
            'C*x1**4+C*x1**3+C*x2**2+C*x2+C',
            # AI Faynman
            'C*exp(C*x1**2+C)/sqrt(C*x2+C)+C',
            'C*x1*x2+C',
            'C*1/2*x1*x2**2+C',
            'C*x1/x2+C',
            'C*arcsin(C*x1*sin(C*x2+C)+C)+C',
            'C*(C*x1/(2*pi)+C)*x2+C',
            'C*3/2*x1*x2+C',
            'C*x1/(C*4*pi*x2**2+C)+C',
            'C*x1*x2**2/2+C',
            'C*1+C*x1*x2/(C*1-C*(C*x1*x2/3+C)+C)+C',
            'C*x1*x2**2+C',
            'C*x1/(2*(1+C*x2+C))+C',
            'C*x1*(C*x2/(2*pi)+C)+C',
            ], 
        3: [
            # AI Faynman
            'C*exp(C*(x1/x2)**2)/(C*sqrt(2*x3)*x2+C)+C',
            'C*x1/sqrt(1-x2**2/x3**2+C)+C',
            'C*x1*x2*x3+C',
            'C*x1*x2/sqrt(C*1-C*x2**2/x3**2+C)+C',
            'C*(C*x1+C*x2+C)/(C*1+C*x1*x2/x3**2+C)+C',
            'C*x1*x3*sin(C*x2+C)+C',
            'C*1/(C*1/x1+C*x2/x3+C)+C',
            'C*x1*sin(C*x2*x3/2+C)**2/sin(x3/2)**2+C',
            'C*arcsin(C*x1/(C*x2*x3+C)+C)+C',
            'C*x1/(C*1-C*x2/x3+C)+C',
            'C*(1+C*x1/x3+C)/sqrt(1-C*x1**2/x3**2+C)*x2+C',
            'C*(C*x1/(C*x3+C)+C)*x2+C',
            'C*x1+C*x2+C*2*sqrt(x1*x2)*cos(x3)+C',
            'C*1/(x1-1)*x2*x3+C',
            'C*x1*x2*x3+C',
            'C*sqrt(x1*x2/x3)+C',
            'C*x1*x2**2/sqrt(C*1-C*x3**2/x2**2+C)+C',
            'C*x1/(C*4*pi*x2*x3+C)+C',
            'C*1/(C*4*pi*x1+C)*x4*cos(C*x2+C)/x3**2+C',
            'C*3/5*x1**2/(C*4*pi*x2*x3+C)+C',
            'C*x1/x2*1/(1+x3)+C',
            'C*x1/sqrt(C*1-C*x2**2/x3**2+C)+C',
            'C*x1*x2/sqrt(C*1-C*x2**2/x3**2+C)+C',
            '-C*x1*x3*COS(C*x2+C)+C',
            '-C*x1*x2*COS(C*x3+C)+C',
            'C*sqrt(C*x1**2/x2**2-C*pi**2/x3**2+C)+C',
            'C*x1*x2*x3**2+C',
            'C*x1*x2/(C*2*pi*x3+C)+C',
            'C*x1*x2*x3/2+C',
            'C*x1*x2/(4*pi*x3)+C',
            'C*x1*(1+C*x2+C)*x3+C',
            'C*2*x1*x2/(C*x3/(2*pi)+C)+C',
            'C*sin(C*x1*x2/(C*x3/(2*pi)+C)+C)**2+C',
            'C*2*x1*(1-C*cos(C*x2*x3+C)+C)+C',
            'C*(C*x1/(2*pi)+C)**2/(C*2*x2*x3**2+C)+C',
            'C*2*pi*x3/(C*x1*x2+C)+C',
            'C*x1*(1+C*x2*cos(x3)+C)+C',
        ], 
        4: [
            # AI Faynman
            'C*exp(C*((C*x1+C*x2+C)/x3)**2+C)/(C*sqrt(C*x4+C)*x3+C)+C', 
            'C*sqrt(C*(C*x2+C*x1+C)**2+(C*x3+C*x4+C)**2+C)+C',    
            'C*x1*x2/(C*x3*x4*x2**3+C)+C',
            'C/2*x1*(C*x2**2+C*x3**2+C*x4**2+C)+C',
            'C*(C*x1-C*x2*x3+C)/sqrt(C*1-C*x2**2/x4**2+C)+C',
            'C*(C*x1-C*x3*x2/x4**2+C)/sqrt(C*1-C*x3**2/x4**2+C)+C',
            'C*(C*x1*x3+C*x2*x4+C)/(C*x1+C*x2+C)+C',
            'C*x1*x2*x3*sin(C*x4+C)+C',
            'C*1/2*x1*(C*x3**2+C*x4**2+C)*1/2*x2**2+C',
            'C*sqrt(C*x1**2+C*x2**2-C*2*x1*x2*cos(C*x3-C*x4+C))+C',
            'C*x1*x2*x3/x4+C',
            'C*4*pi*x1*(C*x2/(2*pi)+C)**2/(C*x3*x4**2+C)+C',
            'C*x1*x2*x3/x4+C',
            'C*1/(C*x1-1+C)*x2*x3/x4+C',
            'C*x1*(C*cos(C*x2*x3+C)+C*x4*cos(C*x2*x3+C)**2+C)+C',
            'C*x1/(C*4*pi*x2+C)*3*cos(C*x3+C)*sin(C*x3+C)/x4**3+C',
            'C*x1*x2/(C*x3*(C*x4**2-x5**2+C)+C)+C',
            'C*x1*x2/(C*1-C*(C*x1*x2/3+C)+C)*x3*x4+C',
            'C*1/(C*4*pi*x1*x2**2+C)*2*x3/x4+C',
            'C*x1*x2*x3/(2*x4)+C',
            'C*x1*x2*x3/x4+C',
            'C*1/(C*exp(C*(C*x1/(2*pi)+C)*x4/(C*x2*x3+C)+C)-1)+C',
            'C*(x1/(2*pi))*x2/(C*exp(C*(C*x1/(2*pi)+C)*x2/(C*x3*x4+C))-1)+C',
            'C*x1*sqrt(C*x2**2+C*x3**2+C*x4**2+C)+C',
            'C*2*x1*x2**2*x3/(C*x4/(2*pi)+C)+C',
            'C*x1*(C*exp(C*x3*x2/(C*x4*x5+C)+C)-1)+C',
            '-C*x1*x2*x3/x4+C',
        ], 
        5: [
            # AI Faynman
            'C*x1*x2*x3/(C*x4*x5*x3**3+C)+C',  
            'C*x1*(C*x2+C*x3*x4*sin(C*x5+C))+C',     
            'C*x1*x2*x3*(C*1/x4-C*1/x5+C)+C',  
            'C*x1/(2*pi)*x2**3/(pi**2*x5**2*(exp((x1/(2*pi))*x2/(x3*x4))-1))+C',   
            'C*x1*x2*x3*ln(x4/x5)+C',
            'C*x1*(C*x2-C*x3+C)*x4/x5+C',
            'C*x1*x2**2*x3/(C*3*x4*x5+C)+C',
            'C*x1/(C*4*pi*x2*x3*(1-C*x4/x5+C)+C)+C',
            'C*x1*x2*x3*x4/(C*x5/(2*pi)+C)+C',
            'C*x1/(C*exp(C*x2*x3/(C*x4*x5+C)+C)+C*exp(-C*x2*x3/(C*x4*x5+C)))+C',
            'C*x1*x2*tanh(C*x2*x3/(C*x4*x5+C)+C)+C',
            '-C*x1*x3**4/(C*2*(C*4*pi*x2+C)**2*(C*x4/(2*pi)+C)**2)*(C*1/x5**2+C)',
        ], 
        6: [
            # AI Faynman
            'C*x1*x4+C*x2*x5+C*x3*x6+C', 
            'C*x1**2*x2**2/(C*6*x3*x4*x5**3+C)+C',     
            'C*x1*exp(-C*x2*x3*x4/(C*x5*x6+C))+C',      
            'C*x1/(C*4*pi*x2+C)*3*x5/x6**5*sqrt(C*x3**2+x4**2+C)+C',
            'C*x1*(1+C*x2*x3*cos(C*x4+C)/(C*x5*x6+C)+C)+C',
            'C*(C*x1*x5*x4/(C*x6/(2*pi)+C)+C)*sin(C*(C*x2-C*x3+C)*x4/2)**2/(C*(C*x2-C*x3+C)*x4/2)**2+C',
        ], 
        7: [
            # AI Faynman
            'C*(C*1/2*x1*x4*x5**2+C)*(C*8*x6*x7**2/3+C)*(C*x2**4/(C*x2**2-C*x3**2+C)**2+C)+C',
            
        ], 
        8: [
            # AI Faynman
            'C*x1*x8/(C*x4*x5+C)+C*(C*x1*x2+C)/(C*x3*x7**2*x4*x5+C)*x6+C',            
        ], 
        9: [
            # AI Faynman
            'C*x3*x4*x5/((C*x2+C*x1+C)**2+(C*x6+C*x7+C)**2+(C*x8+C*x9)**2+C)+C',
        ], 
    }

# Check if there is a similar equation in the training set

# eqList = [element for element in templatesEQs.values()] #list(NGUYEN2EqTemplates.values())
eqList = []
for lst in templatesEQs.values():
    eqList.extend(lst)

bestErr = {key:0 for key in eqList}
bestEquations = {'C':0 for key in eqList}
bestTemplates = {'C':0 for key in eqList}

print('The list of equations:', eqList)
foundEQ = {}
for sample in tqdm(trainText):
    try:
        sample = json.loads(sample) # convert the sequence tokens to a dictionary
    except:
        print("Couldn't convert to json: {}".format(sample))
        idx = idx - 1 
        idx = idx if idx>=0 else 0
        sample = trainText[idx]
        sample = json.loads(sample)
        
    # find the number of variables in the equation
    eq = sample['Skeleton']

    if eq in eqList:
        foundEQ[eq] = 1 if eq not in foundEQ else foundEQ[eq] + 1
        print('This equation has been found in the data: {}'.format(eq))

# save the results to a file
fileName = './benchmarkSimilarEquations.json'
with open(fileName, 'w', encoding="utf-8") as o:
    json.dump(foundEQ, o)

for eq in eqList:
    if eq not in foundEQ.keys():
        print('This equation: {} is not in the data!'.format(eq))
        
print('{} has been saved succesfully.'.format(fileName))

# load the best model
model.load_state_dict(torch.load(ckptPath))
model = model.eval().to(trainer.device)
print('{} model has been loaded!'.format(ckptPath))

numTests = 100
from utils import *
for i in tqdm(range(numTests)):
    for skeletonEqn in eqList:
        # replace the constants with new ones
        target = ''
        for chr in skeletonEqn:
            if chr == 'C':
                # genereate a new random number
                chr = '{}'.format(np.random.uniform(const_range[0], const_range[1]))
            target += chr
        pointsList = []
        pointsListTest = []
        for p in range(numPoints):
            minX, maxX = (trainRange[0], trainRange[1]) 
            x = list(np.round(np.random.uniform(minX, maxX, numVars), decimals))
            tmpEq = target + ''
            for nVID in range(numVars):
                tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))

            try:
                y = float(np.round(eval(tmpEq), decimals))
            except:
                # ZeroDivisionError: 0.0 cannot be raised to a negative power
                y = 0

            p = (x,y)
            #print('p:{}, range:{}'.format(p, (minX, maxX)))
            pointsList.append(p)

        # generate test points
        for p in range(numPoints):
            minX, maxX = (testRange[0][0], testRange[1][0]) if random.random() < 0.5 else (testRange[0][1], testRange[1][1])
            x = list(np.round(np.random.uniform(minX, maxX, numVars), decimals))
            tmpEq = target + ''
            for nVID in range(numVars):
                tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))

            y = float(np.round(eval(tmpEq), decimals))
            p = (x,y)
            pointsListTest.append(p)

        # initialized the input variable with start token <
        inputs = torch.tensor([[train_dataset.stoi['<']]]).to(trainer.device)

        # extract points from the input sequence
        points = torch.zeros(numVars+numYs, numPoints)
        for idx, xy in enumerate(pointsList): # zip(X,Y)): #
            x = xy[0][:numVars] + [0]*(max(numVars-len(xy[0]),0)) # padding
            y = [xy[1]] if type(xy[1])==float or type(xy[1])==int else xy[1]

            y = y + [0]*(max(numYs-len(y),0)) # padding

            p = x+y # because it is only one point 
            p = torch.tensor(p)
            #replace nan and inf
            p = torch.nan_to_num(p, nan=0.0, 
                                    posinf=train_dataset.threshold[1], 
                                    neginf=train_dataset.threshold[0])
            # p[p>train_dataset.threshold[1]] = train_dataset.threshold[1] # clip the upper bound
            # p[p<train_dataset.threshold[0]] = train_dataset.threshold[0] # clip the lower bound
            points[:,idx] = p
        
        # Normalize points between zero and one # DxN
        # eps = 1e-5
        # minP = points.min(dim=1, keepdim=True)[0]
        # maxP = points.max(dim=1, keepdim=True)[0]
        # points -= minP
        # points /= (maxP-minP+eps) 
        # points = torch.nan_to_num(points, nan=train_dataset.threshold[1], 
        #                         posinf=train_dataset.threshold[1], 
        #                         neginf=train_dataset.threshold[0])
        points -= points.mean()
        points /= points.std()

        points = points.unsqueeze(0).to(trainer.device)
        outputsHat = sample(model, inputs, blockSize, points=points,
                        temperature=1.0, sample=True, #top_k=40
                        )[0]

        # filter out predicted
        predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])
        predicted = predicted.strip(train_dataset.paddingToken).split('>')
        predicted = predicted[0] if len(predicted[0])>=1 else predicted[1]
        predicted = predicted.strip('<').strip(">")
        #predicted = predicted[1:] # ignore the number of variables in the equation (first token)

        predicted = predicted.replace(' ','')
        predicted = predicted.replace('\n','')
        template = predicted + ''
        target = target.replace(' ','')
        target = target.replace('\n','')

        # extract points from the input sequence
        pointsTest = torch.zeros(numVars+numYs, numPoints)
        for idx, xy in enumerate(pointsListTest):
            x = xy[0][:numVars] + [0]*(max(numVars-len(xy[0][:numVars]),0)) # padding
            y = [xy[1]] if type(xy[1])==float else xy[1]
            y = y + [0]*(max(numYs-len(y),0)) # padding

            p = x+y # because it is only one point 
            p = torch.tensor(p)
            #replace nan and inf
            p = torch.nan_to_num(p, nan=0.0, 
                                    posinf=train_dataset.threshold[1], 
                                    neginf=train_dataset.threshold[0])
            # p[p>train_dataset.threshold[1]] = train_dataset.threshold[1] # clip the upper bound
            # p[p<train_dataset.threshold[0]] = train_dataset.threshold[0] # clip the lower bound
            pointsTest[:,idx] = p

        # Normalize points between zero and one # DxN
        # eps = 1e-5
        # minP = pointsTest.min(dim=1, keepdim=True)[0]
        # maxP = pointsTest.max(dim=1, keepdim=True)[0]
        # pointsTest -= minP
        # pointsTest /= (maxP-minP+eps) 
        # pointsTest = torch.nan_to_num(pointsTest, nan=train_dataset.threshold[1], 
        #                         posinf=train_dataset.threshold[1], 
        #                         neginf=train_dataset.threshold[0])
        pointsTest = pointsTest.numpy()

        # optimize the constants
        # train a regressor to find the constants (too slow)
        c = [1 for i,x in enumerate(predicted) if x=='C']   # variables
        c[-1] = 0
        # b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
        # predicted = predicted.replace('C','{}').format(*c)
        try:
            if len(c) == 0:
                print('No constants in the equation. Eq:{}'.format(predicted))
                pass # do nothing
            else:
                # for easier comparison, we are using minimize package  
                cHat = minimize(lossFunc, c, #method='BFGS', #bounds=b, 
                                args=(predicted, points[:,:numVars,:].squeeze().cpu().T.numpy(), points[:,numVars:,:].squeeze().cpu().T.numpy())) 
                predicted = predicted.replace('C','{}').format(*cHat.x)
        except:
            print('Wrong Equation:{}'.format(predicted))
            raise
            predicted = 0

        Ys = [] 
        Yhats = []
        for xs in pointsTest[:-1,:numPoints].T:
            try:
                eqTmp = target + '' # copy eq
                for i,x in enumerate(xs):
                    # replace xi with the value in the eq
                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                    if ',' in eqTmp:
                        assert 'There is a , in the equation!'

                #print('target',eqTmp)
                YEval = eval(eqTmp)
                # YEval = 0 if np.isnan(YEval) else YEval
                # YEval = 100 if np.isinf(YEval) else YEval
            except:
                print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
            Ys.append(YEval)
            try:
                eqTmp = predicted + '' # copy eq
                for i,x in enumerate(xs):
                    # replace xi with the value in the eq
                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                    if ',' in eqTmp:
                        assert 'There is a , in the equation!'
                Yhat = eval(eqTmp)
                # Yhat = 0 if np.isnan(Yhat) else Yhat
                # Yhat = 1000 if np.isinf(Yhat) else Yhat
            except:
                print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                Yhat = 1000
            Yhats.append(Yhat)

        err = relativeErr(Ys,Yhats, info=True)
        
        if err < bestErr[skeletonEqn]:
            bestErr[skeletonEqn] = err
            bestEquations[skeletonEqn] = predicted
            bestTemplates[skeletonEqn] = template

        print('\nTemplate:{}\nTarget:{}\nPredicted:{}\ntemplate:{}\nErr:{}\n'.format(skeletonEqn, target, predicted, template, err))
    
# print the final equations
print('\n\n','-'*100)
for pr,te,ta,er in zip(bestEquations, bestTemplates, eqList, bestErr):
    print('PR:{}\nTE:{}\nTA:{}\nErr:{}\n'.format(bestEquations[pr],bestTemplates[te],eqList[ta],bestErr[er]))