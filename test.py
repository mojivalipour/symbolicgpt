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

# NOTE: make sure your data points are in the same range with trainRange

# config
numEpochs = 4 # number of epochs to train the GPT+PT model
scratch=True # if you want to ignore the cache and start for scratch
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=[200,201] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=2 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 400 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
decimals = 8 # decimals of the points only if target is Skeleton
trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton
testRange = [[-5.0, 3.0],[-3.0, 5.0]]
numTests = 1 # number of times to generate candidates for one test equation
useRange = True
dataDir = 'D:/Datasets/Symbolic Dataset/Datasets/FirstDataGenerator/' #'./datasets/'
dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize'.format(numVars, numPoints[0], numPoints[1], embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '2Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_200Points'
dataTestFolder = '2Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_200Points'
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
test_dataset = CharDataset(textTest, blockSize, chars, numVars=numVars, 
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

## Test the model
# alright, let's sample some character-level symbolic GPT 
loader = torch.utils.data.DataLoader(
                                test_dataset, 
                                shuffle=False, 
                                pin_memory=True,
                                batch_size=1,
                                num_workers=0)

from utils import *
resultDict = {}
try:
    with open(fName, 'w', encoding="utf-8") as o:
        resultDict[fName] = {'SymbolicGPT':[]}

        for i, batch in enumerate(loader):

            inputs,outputs,points,variables = batch

            eq = ''.join([train_dataset.itos[int(i)] for i in outputs[0]])
            eq = eq.strip(train_dataset.paddingToken).split('>')
            eq = eq[0] #if len(eq[0])>=1 else eq[1]
            eq = eq.strip('<').strip(">")

            print('Test Case {}.'.format(i))
            o.write('Test Case {}/{}.\n'.format(i,len(textTest)-1))

            t = json.loads(textTest[i])

            inputs = inputs[:,0:1].to(trainer.device)
            points = points.to(trainer.device)
            # points = points[:,:numPoints] # filter anything more than maximum number of points
            variables = variables.to(trainer.device)

            bestErr = 10000000
            bestPredicted = 'C'
            for i in tqdm(range(numTests)):
                outputsHat = sample(model, 
                                    inputs, 
                                    blockSize, 
                                    points=points,
                                    variables=variables,
                                    temperature=0.9, 
                                    sample=True, 
                                    top_k=40,
                                    top_p=0.7,
                                    )[0]

                # filter out predicted
                predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])

                if variableEmbedding == 'STR_VAR':
                    eq = eq.split(':')[-1]
                    predicted = predicted.split(':')[-1]

                predicted = predicted.strip(train_dataset.paddingToken).split('>')
                predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
                predicted = predicted.strip('<').strip(">")
                predicted = predicted.replace('Ce','C*e')

                print('Target:{}\nSkeleton:{}'.format(eq, predicted))

                # train a regressor to find the constants (too slow)
                c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
                # c[-1] = 0 # initialize the constant as zero
                b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
                try:
                    if len(c) != 0:
                        # This is the bottleneck in our algorithm
                        # for easier comparison, we are using minimize package  
                        cHat = minimize(lossFunc, c, #bounds=b,
                                    args=(predicted, t['X'], t['Y'])) 
            
                        predicted = predicted.replace('C','{}').format(*cHat.x)
                except ValueError:
                    raise 'Err: Wrong Equation {}'.format(predicted)
                except Exception as e:
                    raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)

                print('Skeleton+LS:{}'.format(predicted))

                Ys = [] #t['YT']
                Yhats = []
                for xs in t['XT']:
                    try:
                        eqTmp = eq + '' # copy eq
                        eqTmp = eqTmp.replace(' ','')
                        eqTmp = eqTmp.replace('\n','')
                        for i,x in enumerate(xs):
                            # replace xi with the value in the eq
                            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                            if ',' in eqTmp:
                                assert 'There is a , in the equation!'
                        YEval = eval(eqTmp)
                        # YEval = 0 if np.isnan(YEval) else YEval
                        # YEval = 100 if np.isinf(YEval) else YEval
                    except:
                        print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                        print(i)
                        raise
                        continue # if there is any point in the target equation that has any problem, ignore it
                        YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
                    Ys.append(YEval)
                    try:
                        eqTmp = predicted + '' # copy eq
                        eqTmp = eqTmp.replace(' ','')
                        eqTmp = eqTmp.replace('\n','')
                        for i,x in enumerate(xs):
                            # replace xi with the value in the eq
                            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                            if ',' in eqTmp:
                                assert 'There is a , in the equation!'
                        Yhat = eval(eqTmp)
                        # Yhat = 0 if np.isnan(Yhat) else Yhat
                        # Yhat = 100 if np.isinf(Yhat) else Yhat
                    except:
                        print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
                        Yhat = 100
                    Yhats.append(Yhat)
                err = relativeErr(Ys,Yhats, info=True)

                if type(err) is np.complex128 or np.complex:
                    err = abs(err.real)

                if err < bestErr:
                    bestErr = err
                    bestPredicted = predicted

                print('Err:{}'.format(err))
                print('') # just an empty line
            
            resultDict[fName]['SymbolicGPT'].append(bestErr)

            o.write('{}\n'.format(eq))
            o.write('{}:\n'.format('SymbolicGPT'))
            o.write('{}\n'.format(bestPredicted))

            o.write('{}\n{}\n\n'.format( 
                                    bestPredicted,
                                    bestErr
                                    ))

    print('Avg Err:{}'.format(np.mean(resultDict[fName]['SymbolicGPT'])))
    
except KeyboardInterrupt:
    print('KeyboardInterrupt')

# plot the error frequency for model comparison
num_eqns = len(resultDict[fName]['SymbolicGPT'])
num_vars = pconf.numberofVars
title = titleTemplate.format(num_eqns, num_vars)

models = list(key for key in resultDict[fName].keys() if len(resultDict[fName][key])==num_eqns)
lists_of_error_scores = [resultDict[fName][key] for key in models if len(resultDict[fName][key])==num_eqns]
linestyles = ["-","dashdot","dotted","--"]

eps = 0.00001
y, x, _ = plt.hist([np.log([max(min(x+eps, 1e5),1e-5) for x in e]) for e in lists_of_error_scores],
                   label=models,
                   cumulative=True, 
                   histtype="step", 
                   bins=2000, 
                   density=True,
                   log=False)
y = np.expand_dims(y,0)
plt.figure(figsize=(15, 10))

for idx, m in enumerate(models): 
    plt.plot(x[:-1], 
           y[idx] * 100, 
           linestyle=linestyles[idx], 
           label=m)

plt.legend(loc="upper left")
plt.title(title)
plt.xlabel("Log of Relative Mean Square Error")
plt.ylabel("Normalized Cumulative Frequency")

name = '{}.png'.format(fName.split('.txt')[0])
plt.savefig(name)