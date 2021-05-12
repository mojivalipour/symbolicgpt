#!/usr/bin/env python
# coding: utf-8

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# make deterministic
from mingpt.utils import set_seed
set_seed(42)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset


# config
embeddingSize=512
numPoints=30
numVars=1
numYs=1
paddingToken=-100
padId=0
block_size = 50 # spatial extent of the model for its context

class CharDataset(Dataset):

    def __init__(self, data, block_size, extractAtt=False):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate([paddingToken]+chars) }
        self.itos = { i:ch for i,ch in enumerate([paddingToken]+chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.attributes = extractAtt
        self.threshold = [-1000,1000]
        
        if self.attributes:
            self.dataList = self.data.split('\n') #TODO: remove later?

            self.blockIdx = []
            summation = 0
            for d in self.dataList:
                s = summation
                e = s + len(d)
                self.blockIdx.append((s,e))
                summation = e+1
    
    def __len__(self):
        if self.attributes:
            return len(self.dataList) - 1
        else:
            return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        #chunk = self.data[idx:idx + self.block_size + 1]
        chunk = self.data[self.blockIdx[idx][0]:self.blockIdx[idx][1]]
        
        # extracts other attributes
        points = None
        if self.attributes:
            dic = json.loads(chunk)
            points = []
            for xy in zip(dic['X'], dic['Y']):
                x = xy[0] + [paddingToken]*(max(numVars-len(xy[0]),0)) # padding
                y = [xy[1]] if type(xy[1])== float else xy[1]
                y = y + [paddingToken]*(max(numYs-len(y),0)) # padding
                
                p = x + y #x.extend(y)
                p = torch.tensor(p)
                
                #replace nan and inf
                p = torch.nan_to_num(p, nan=0.0, 
                                     posinf=self.threshold[1], 
                                     neginf=self.threshold[0])

                points.append(p)
            chunk = '"'+dic['EQ']+'"'
        
        # encode every character to an integer
        dix = [self.stoi[s] for i,s in enumerate(chunk) if i<self.block_size]
        paddingSize = max(self.block_size-len(dix),0)
        
        mask = [1 for s in dix]
        dixX = dix + [self.stoi[paddingToken]]*paddingSize # padding
        dix += [paddingToken]*paddingSize # padding
        mask += [0]*paddingSize
        
        inputs = torch.tensor(dixX[:-1], dtype=torch.long).contiguous()
        mask = torch.tensor(mask[:-1], dtype=torch.long).contiguous()
        
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """        
        
        outputs = torch.tensor(dix[1:], dtype=torch.long).contiguous()
        
        assert mask.shape==outputs.shape==inputs.shape, 'M:{}-O:{}-I:{}'.format(mask.shape,outputs.shape,inputs.shape)
        assert len(mask) == self.block_size-1, 'Wrong mask shape: {}'.format(mask.shape)
        assert len(inputs) == self.block_size-1, 'Wrong inputs shape: {}'.format(inputs.shape)
        assert len(outputs) == self.block_size-1, 'Wrong y shape: {}'.format(outputs.shape)
        assert len(points) == numPoints, 'Wrong #points: {}'.format(len(points))
        
        return inputs, outputs, points, mask

import json
from tqdm import tqdm
import glob
def processDataFiles(files):
    text = ''""
    for f in tqdm(files):
        with open(f, 'r') as h: 
            lines = h.read() # don't worry we won't run out of file handles
            text += lines #json.loads(line)                
    return text

path = 'D:\Datasets\Symbolic Dataset\Datasets\Mesh_Simple_GPT2_Sorted\TrainDatasetFixed\*.json'
files = glob.glob(path)
text = processDataFiles([files[0]])

train_dataset = CharDataset(text, block_size, extractAtt=True) # one line of poem is roughly 50 characters

idx = np.random.randint(train_dataset.__len__())
sample = train_dataset.__getitem__(idx)
x,y,p,m = sample
print('XS:{}\nMS:{}\nyS:{}\nPointsS:{}'.format(x.shape,m.shape,y.shape,len(p)))
print('X:{}\nM:{}\ny:{}\nPoints:{}'.format(x,m,y,p))


path = 'D:\Datasets\Symbolic Dataset\Datasets\Mesh_Simple_GPT2_Sorted\TestDataset\*.json'
files = glob.glob(path)
textTest = processDataFiles([files[0]])
test_dataset = CharDataset(textTest, block_size, extractAtt=True)

from mingpt.model import GPT, GPTConfig, PointNetConfig
pconf = PointNetConfig(embeddingSize=embeddingSize, 
                       numberofPoints=numPoints, 
                       numberofVars=numVars, 
                       numberofYs=numYs)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=embeddingSize, grad_norm_clip=1.0,
                  padToken=paddingToken, padId=padId)
model = GPT(mconf, pconf)


from mingpt.trainer import Trainer, TrainerConfig

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=2, batch_size=2, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset, tconf)

try:
    trainer.train()
except KeyboardInterrupt:
    print('KeyboardInterrupt')

# add a safe wrapper for numpy math functions
from numpy import *
import numpy as np

def divide(x, y):
  x = np.nan_to_num(x)
  y = np.nan_to_num(y)
  return np.divide(x,y+1e-5)

def sqrt(x):
  x = np.nan_to_num(x)
  return np.sqrt(np.abs(x)) 

# Mean square error
def mse(y, y_hat):
    y_hat = np.reshape(y_hat, [1, -1])[0]
    y_gold = np.reshape(y, [1, -1])[0]
    our_sum = 0
    for i in range(len(y_gold)):
        our_sum += (y_hat[i] - y_gold[i]) ** 2

    return our_sum / len(y_gold)

# alright, let's sample some character-level symbolic GPT
from mingpt.utils import sample
from gp_model import Genetic_Model
from mlp_model import MLP_Model
    
loader = torch.utils.data.DataLoader(
                                test_dataset, 
                                shuffle=False, 
                                pin_memory=True,
                                batch_size=1,
                                num_workers=4)

testRange = [3.1,6.0]
numTestPoints = 10
#test = np.linspace(3.1,6.0,numTestPoints)

# gpm = Genetic_Model(n_jobs=-1)
# mlp = MLP_Model()
    
fName = 'res.txt'
resultDict = {}
with open(fName, 'w', encoding="utf-8") as o:
    textTestList = textTest.split('\n')
    modelName = 'SymbolicGPT'
    resultDict[fName] = {modelName:[]}
    
    for i, (x,y,p) in enumerate(loader):
        
        print('Test Case {}.'.format(i))
        o.write('Test Case {}/{}.'.format(i,len(textTestList)))
        
        t = json.loads(textTestList[i])
        x = x[:,0:1].to(trainer.device)
        p = [x.to(trainer.device) for x in p]
        yHat = sample(model, x, 50, points=p, 
                      temperature=1.0, sample=True, 
                      top_k=10)[0]
        
        target = ''.join([train_dataset.itos[int(i)] for i in y[0]]).strip('"')
        o.write('{}'.format(target))
        
        predicted = ''.join([train_dataset.itos[int(i)] for i in yHat])
        # filter out predicted
        predicted = predicted.split('"')[1]
        
        print('Target:{}\nPredicted:{}'.format(target, predicted))
        
        Ys = [] #t['YT']
        Yhats = []
        for xs in t['XT']:
            try:
                eqTmp = target + '' # copy eq
                eqTmp = eqTmp.replace(' ','')
                eqTmp = eqTmp.replace('\n','')
                for i,x in enumerate(xs):
                    # replace xi with the value in the eq
                    eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                    if ',' in eqTmp:
                        assert 'There is a , in the equation!'
                YEval = eval(eqTmp)
                YEval = 0 if np.isnan(YEval) else YEval
                YEval = 10000 if np.isinf(YEval) else YEval
            except:
                YEval = 0
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
                Yhat = 0 if np.isnan(Yhat) else Yhat
                Yhat = 10000 if np.isinf(Yhat) else Yhat
            except:
                Yhat = 0
            Yhats.append(Yhat)
        mseErr = mse(Ys,Yhats)
        
        if type(mseErr) is np.complex128:
            mseErr = mseErr.real
        elif mseErr < 0.00005:
            mseErr = 0
            
        resultDict[fName][modelName].append(mseErr)
        
        o.write('{}:{}\n{}'.format(modelName, 
                               mseErr,
                               predicted))
        
        print('MSE:{}\n'.format(mseErr))

# plot the error frequency for model comparison
from matplotlib import pyplot as plt
num_eqns = len(resultDict[fName]['SymbolicGPT'])
num_vars = pconf.numberofVars

models = list(resultDict[fName].keys())
lists_of_error_scores = [resultDict[fName][key] for key in models]
linestyles = ["-","dashdot","dotted","--"]

eps = 0.00001
y, x, _ = plt.hist([np.log([x+eps for x in e]) for e in lists_of_error_scores],
                   label=models,
                   cumulative=True, 
                   histtype="step", 
                   bins=2000, 
                   density="true")
y = np.expand_dims(y,0)
plt.figure(figsize=(15, 10))

for idx, model in enumerate(models): 
    plt.plot(x[:-1], 
           y[idx] * 100, 
           linestyle=linestyles[idx], 
           label=model)

plt.legend(loc="upper left")
plt.title("{} equations of {} variables".format(num_eqns, num_vars))
plt.xlabel("Log of Mean Square Error")
plt.ylabel("Normalized Cumulative Frequency")

name = '{}.png'.format('results')
plt.savefig(name)


