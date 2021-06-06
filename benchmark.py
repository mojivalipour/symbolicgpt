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
import math
import random
import numpy as np
from tqdm import tqdm
from numpy import * # to override the math functions

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

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
numPoints=20 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=2 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 100 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_1-{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of 1-{} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1-2Var_RandSupport_RandLength_-1to1_-5to1_1to5_100to500Points'
addr = './SavedModels/' # where to save model
method = 'EMB_CON' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
# EMB_CAT: Concat point embedding to GPT token+pos embedding
# EMB_SUM: Add point embedding to GPT tokens+pos embedding
# OUT_CAT: Concat the output of the self-attention and point embedding
# OUT_SUM: Add the output of the self-attention and point embedding
# EMB_CON: Conditional Embedding, add the point embedding as the first token
maxNumFiles = 30 # maximum number of file to load in memory for training the neural network
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_SymbolicGPT_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                             'GPT_PT_{}_{}'.format(method, target), 
                                             'Padding',
                                             blockSize)
ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
try: 
    os.mkdir(addr)
except:
    print('Folder already exists!')

# load the train dataset
path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
files = glob.glob(path)[:maxNumFiles]
text = processDataFiles(files)
chars = sorted(list(set(text))+['_','T','<','>']) # extract unique characters from the text before converting the text to a list, # T is for the test data
text = text.split('\n') # convert the raw text to a set of examples
text = text[:-1] if len(text[-1]) == 0 else text
random.shuffle(text) # shuffle the dataset, it's important for combined number of variables
train_dataset = CharDataset(text, blockSize, chars, numVars=numVars, numYs=numYs, numPoints=numPoints, target=target) 

# print a random sample
idx = np.random.randint(train_dataset.__len__())
inputs, outputs, points = train_dataset.__getitem__(idx)
print('inputs:{}'.format(inputs))
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}'.format(idx,inputs,outputs,points))

# load the val dataset
path = '{}/{}/Val/*.json'.format(dataDir,dataFolder)
files = glob.glob(path)
textVal = processDataFiles([files[0]])
textVal = textVal.split('\n') # convert the raw text to a set of examples
val_dataset = CharDataset(textVal, blockSize, chars, numVars=numVars, numYs=numYs, numPoints=numPoints, target=target)

# print a random sample
idx = np.random.randint(val_dataset.__len__())
inputs, outputs, points = val_dataset.__getitem__(idx)
print(points.min(), points.max())
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}'.format(idx,inputs,outputs,points))

# load the test data
path = '{}/{}/Test/*.json'.format(dataDir,dataFolder)
files = glob.glob(path)
textTest = processDataFiles(files)
textTest = textTest.split('\n') # convert the raw text to a set of examples
# test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
test_dataset = CharDataset(textTest, blockSize, chars, numVars=numVars, numYs=numYs, numPoints=numPoints)

# print a random sample
idx = np.random.randint(test_dataset.__len__())
inputs, outputs, points = test_dataset.__getitem__(idx)
print(points.min(), points.max())
inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
print('id:{}\ninputs:{}\noutputs:{}\npoints:{}'.format(idx,inputs,outputs,points))

# create the model
pconf = PointNetConfig(embeddingSize=embeddingSize, 
                       numberofPoints=numPoints, 
                       numberofVars=numVars, 
                       numberofYs=numYs,
                       method=method)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize, padding_idx=train_dataset.paddingID)
model = GPT(mconf, pconf)
    
# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*blockSize,
                      num_workers=0, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss)

# load the best model
model.load_state_dict(torch.load(ckptPath))
model = model.eval().to(trainer.device)
print('{} model has been loaded!'.format(ckptPath))

# benchmarks
import csv

rng = np.random.RandomState(seed)
benchmarkPath = './benchmark/dsr-benchmark-data/'
dataPoints = glob.glob(benchmarkPath+'*.csv')
NGUYEN2Eq = {
           '1':'x1**3+x1**2+x1',
           '2':'x1**4+x1**3+x1**2+x1',
           '3':'x1**5+x1**4+x1**3+x1**2+x1',
           '4':'x1**6+x1**5+x1**4+x1**3+x1**2+x1',
           '5':'sin(x1**2)*cos(x1)-1',
           '6':'sin(x1)+sin(x1+x1**2)',
           '7':'log(x1+1)+log(x1**2+1)',
           '8':'sqrt(x1)',
           '9':'sin(x1)+sin(x2**2)',
           '10':'2*sin(x1)*cos(x2)',
           '11':'x1**x2',
           '12':'x1**4-x1**3+x2**2/2-x2',}
NGUYENNumVars = {
           '1':1,
           '2':1,
           '3':1,
           '4':1,
           '5':1,
           '6':1,
           '7':1,
           '8':1,
           '9':2,
           '10':2,
           '11':2,
           '12':2,}

bestEquations = {
   '1':'C',
   '2':'C',
   '3':'C',
   '4':'C',
   '5':'C',
   '6':'C',
   '7':'C',
   '8':'C',
   '9':'C',
   '10':'C',
   '11':'C',
   '12':'C',
}

bestErr = {
   '1':1000,
   '2':1000,
   '3':1000,
   '4':1000,
   '5':1000,
   '6':1000,
   '7':1000,
   '8':1000,
   '9':1000,
   '10':1000,
   '11':1000,
   '12':1000,
}

numTests = 10

for i in tqdm(range(numTests)):
    for dataPoint in dataPoints: 
        key = dataPoint.split('\\')[-1].split('Nguyen-')[-1].split('_')[0]
        if not key in NGUYEN2Eq.keys():
            continue
        target = NGUYEN2Eq[key]
        with open(dataPoint, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            pointsList = []
            pointsListTest = []
            for i, c in enumerate(reader):
                p = ([eval(x) for x in c[:-1]],eval(c[-1]))
                if i < numPoints:
                    pointsList.append(p)
                else:
                    pointsListTest.append(p)

            # initialized the input variable with start token <
            inputs = torch.tensor([[train_dataset.stoi['<']]]).to(trainer.device)

            # extract points from the input sequence
            points = torch.zeros(numVars+numYs, numPoints)
            for idx, xy in enumerate(pointsList): # zip(X,Y)): #
                x = xy[0][:numVars] + [0]*(max(numVars-len(xy[0]),0)) # padding
                y = [xy[1]] if type(xy[1])== float else xy[1]

                y = y + [0]*(max(numYs-len(y),0)) # padding

                p = x+y # because it is only one point 
                p = torch.tensor(p)
                #replace nan and inf
                p = torch.nan_to_num(p, nan=0.0, 
                                     posinf=train_dataset.threshold[1], 
                                     neginf=train_dataset.threshold[0])
                p[p>train_dataset.threshold[1]] = train_dataset.threshold[1] # clip the upper bound
                p[p<train_dataset.threshold[0]] = train_dataset.threshold[0] # clip the lower bound
                points[:,idx] = p

            points = points.unsqueeze(0).to(trainer.device)
            outputsHat = sample(model, inputs, blockSize, points=points,
                          temperature=1.0, sample=True, 
                          top_k=40)[0]

            # filter out predicted
            predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])
            predicted = predicted.strip(train_dataset.paddingToken).split('>')
            predicted = predicted[0] if len(predicted[0])>=1 else predicted[1]
            predicted = predicted.strip('<').strip(">")
            #predicted = predicted[1:] # ignore the number of variables in the equation (first token)

            predicted = predicted.replace(' ','')
            predicted = predicted.replace('\n','')
            target = target.replace(' ','')
            target = target.replace('\n','')

            # extract points from the input sequence
            pointsTest = torch.zeros(numVars+numYs, numPoints).numpy()
            for idx, xy in enumerate(pointsListTest):
                x = xy[0][:numVars] + [0]*(max(numVars-len(xy[0]),0)) # padding
                y = [xy[1]] if type(xy[1])== float else xy[1]
                y = y + [0]*(max(numYs-len(y),0)) # padding

                p = x+y # because it is only one point 
                p = torch.tensor(p)
                #replace nan and inf
                p = torch.nan_to_num(p, nan=0.0, 
                                     posinf=train_dataset.threshold[1], 
                                     neginf=train_dataset.threshold[0])
                p[p>train_dataset.threshold[1]] = train_dataset.threshold[1] # clip the upper bound
                p[p<train_dataset.threshold[0]] = train_dataset.threshold[0] # clip the lower bound
                pointsTest[:,idx] = p

            # optimize the constants
            # train a regressor to find the constants (too slow)
            c = [1 for i,x in enumerate(predicted) if x=='C']     
            #predicted = predicted.replace('C','{}').format(*c)
            try:
                if len(c) == 0:
                    print('No constants in the equation. Eq:{}'.format(predicted))
                    pass # do nothing
                else:
                    # for easier comparison, we are using minimize package  
                    cHat = minimize(lossFunc, c,
                                   args=(predicted, points[:,:numVars,:20].squeeze().cpu().T, points[:,numVars:,:20].squeeze().cpu().T)) 
                    predicted = predicted.replace('C','{}').format(*cHat.x)
            except:
                print('Wrong Equation:{}'.format(predicted))
                raise
                predicted = 0

            Ys = [] 
            Yhats = []
            for xs in pointsTest[:-1,:20].T:
                try:
                    eqTmp = target + '' # copy eq
                    for i,x in enumerate(xs):
                        # replace xi with the value in the eq
                        eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                        if ',' in eqTmp:
                            assert 'There is a , in the equation!'
                    #print('target',eqTmp)
                    YEval = eval(eqTmp)
                    YEval = 0 if np.isnan(YEval) else YEval
                    YEval = 100 if np.isinf(YEval) else YEval
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
                    Yhat = 0 if np.isnan(Yhat) else Yhat
                    Yhat = 1000 if np.isinf(Yhat) else Yhat
                except:
                    print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                    Yhat = 1000
                Yhats.append(Yhat)

            err = relativeErr(Ys,Yhats)
            
            if err < bestErr[key]:
                bestErr[key] = err
                bestEquations[key] = predicted

            print('NGUYEN-{} --> Target:{}\nPredicted:{}\nErr:{}\n'.format(key, target, predicted, err))

# print the final equations
print('\n\n',''*100)
for pr,ta,er in zip(bestEquations, NGUYEN2Eq,bestErr):
    print('PR:{}\nTA:{}\nErr:{}\n'.format(bestEquations[pr],NGUYEN2Eq[ta],bestErr[er]))