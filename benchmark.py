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
numPoints=30 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=2 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
trainRange = [-3.0,3.0] 
testRange = [[-5.0, 3.0],[-3.0, 5.0]]
useRange = True
decimals = 8
dataDir = 'D:/Datasets/Symbolic Dataset/Datasets/FirstDataGenerator/' #'./datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1-2Var_RandSupport_RandLength__-3to3_-5.0to-3.0-3.0to5.0_10-30Points'
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
path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
files = glob.glob(path)[:maxNumFiles]
text = processDataFiles(files)
chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
text = text.split('\n') # convert the raw text to a set of examples
text = text[:-1] if len(text[-1]) == 0 else text
trainText = text.copy() # keep a copy of trainText
random.shuffle(text) # shuffle the dataset, it's important for combined number of variables
train_dataset = CharDataset(text, blockSize, chars, numVars=numVars, 
                numYs=numYs, numPoints=numPoints, target=target, addVars=addVars) 

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

NGUYEN2EqTemplates = {
            '1':'C*x1**3+C*x1**2+C*x1+C',
            '2':'C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            '3':'C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            '4':'C*x1**6+C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            '5':'C*sin(C*x1**2)*cos(C*x1+C)+C',
            '6':'C*sin(C*x1+C)+C*sin(C*x1+C*x1**2)+C',
            '7':'C*log(C*x1+C)+C*log(C*x1**2+C)+C',
            '8':'C*sqrt(C*x1+C)+C',
            '9':'C*sin(C*x1+C)+C*sin(C*x2**2+C)+C',
            '10':'C*sin(C*x1+C)*cos(C*x2+C)+C',
            '11':'C*x1**x2+C',
            '12':'C*x1**4+C*x1**3+C*x2**2+C*x2+C',}

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

bestTemplates = {
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

# Check if there is a similar equation in the training set
eqList = list(NGUYEN2EqTemplates.values())
foundEQ = {}
for sample in tqdm(trainText):
    try:
        sample = json.loads(sample) # convert the sequence tokens to a dictionary
    except:
        print("Couldn't convert to json: {}".format(sample))
        
    # find the number of variables in the equation
    eq = sample['Skeleton']

    if eq in eqList:
        foundEQ[eq] = 1 if eq not in foundEQ else foundEQ[eq] + 1
        print('This equation has been found in the data: {}'.format(eq))

# save the results to a file
fileName = './benchmarkSimilarEquations.json'
with open(fileName, 'w', encoding="utf-8") as o:
    json.dump(foundEQ, o)
print('{} has been saved succesfully.'.format(fileName))

# NGUYEN2EqTemplates

numTests = 100
from utils import *
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

            if useRange:
                for p in range(numPoints):
                    minX, maxX = (trainRange[0], trainRange[1]) 
                    x = list(np.round(np.random.uniform(minX, maxX, max(NGUYENNumVars[key],numVars)), decimals))
                    tmpEq = target + ''
                    for nVID in range(max(NGUYENNumVars[key],numVars)):
                        tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))
                    
                    # # if there is still vars in the equation, use zero
                    # for nVID in range(NGUYENNumVars[key]):
                    #     tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(0))

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
                    x = list(np.round(np.random.uniform(minX, maxX, max(NGUYENNumVars[key],numVars)), decimals))
                    tmpEq = target + ''
                    for nVID in range(max(NGUYENNumVars[key],numVars)):
                        tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))

                    # # if there is still vars in the equation, use zero
                    # for nVID in range(NGUYENNumVars[key]):
                    #     tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(0))

                    y = float(np.round(eval(tmpEq), decimals))
                    p = (x,y)
                    #print('#T p:{}, range:{}'.format(p, (minX, maxX)))
                    pointsListTest.append(p)
            else: 
                for i, c in enumerate(reader):
                    p = ([eval(x) for x in c[:-1]],eval(c[-1])) # use the benchmark points                        
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
            pointsTest = torch.zeros(max(NGUYENNumVars[key],numVars)+numYs, numPoints)
            for idx, xy in enumerate(pointsListTest):
                x = xy[0][:numVars] + [0]*(max(max(NGUYENNumVars[key],numVars)-len(xy[0][:numVars]),0)) # padding
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
            b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
            #predicted = predicted.replace('C','{}').format(*c)
            try:
                if len(c) == 0:
                    print('No constants in the equation. Eq:{}'.format(predicted))
                    pass # do nothing
                else:
                    # for easier comparison, we are using minimize package  
                    cHat = minimize(lossFunc, c, #bounds=b, 
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
            
            if err < bestErr[key]:
                bestErr[key] = err
                bestEquations[key] = predicted
                bestTemplates[key] = template

            print('NGUYEN-{} --> Target:{}\nPredicted:{}\ntemplate:{}\nErr:{}\n'.format(key, target, predicted, template, err))

# print the final equations
print('\n\n','-'*100)
for pr,te,ta,er in zip(bestEquations, bestTemplates, NGUYEN2Eq, bestErr):
    print('PR:{}\nTE:{}\nTA:{}\nErr:{}\n'.format(bestEquations[pr],bestTemplates[te],NGUYEN2Eq[ta],bestErr[er]))