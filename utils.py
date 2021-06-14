import re
import json
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn import functional as F
from numpy import * # to override the math functions

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, points=None, variables=None, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond, points=points, variables=variables)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

# helper class and functions
# add a safe wrapper for numpy math functions

def divide(x, y):
  x = np.nan_to_num(x)
  y = np.nan_to_num(y)
  return np.divide(x,max(y,1.0))

def sqrt(x):
  x = np.nan_to_num(x)
  x = np.abs(x)
  return np.sqrt(x) 

def log(x, eps=1e-5):
  x = np.nan_to_num(x)
  x = np.sqrt(x*x+eps)
  return np.log(x)

def exp(x, eps=1e-5):
    x = np.nan_to_num(x)
    #x = np.minimum(x,5) # to avoid overflow
    return np.exp(x)

# Mean square error
def mse(y, y_hat):
    y_hat = np.reshape(y_hat, [1, -1])[0]
    y_gold = np.reshape(y, [1, -1])[0]
    our_sum = 0
    for i in range(len(y_gold)):
        our_sum += (y_hat[i] - y_gold[i]) ** 2

    return our_sum / len(y_gold)

# Relative Mean Square Error
def relativeErr(y, yHat, info=False, eps=1e-5):
    yHat = np.reshape(yHat, [1, -1])[0]
    y = np.reshape(y, [1, -1])[0]
    if len(y) > 0 and len(y)==len(yHat):
        err = ( (yHat - y) )** 2 / np.linalg.norm(y+eps)
        if info:
            for _ in range(5):
                i = np.random.randint(len(y))
                print('yPR,yTrue:{},{}, Err:{}'.format(yHat[i],y[i],err[i]))
    else:
        err = 100

    return np.mean(err)

class CharDataset(Dataset):
    def __init__(self, data, block_size, chars, 
        numVars, numYs, numPoints, target='EQ', addVars=False):
        data_size, vocab_size = len(data), len(chars)
        print('data has %d examples, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        self.numVars = numVars
        self.numYs = numYs
        self.numPoints = numPoints
        
        # padding token
        self.paddingToken = '_'
        self.paddingID = self.stoi[self.paddingToken]
        self.stoi[self.paddingToken] = self.paddingID
        self.itos[self.paddingID] = self.paddingToken
        self.threshold = [-1000,1000]
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data # it should be a list of examples
        self.target = target
        self.addVars = addVars
    
    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        # grab an example from the data
        chunk = self.data[idx] # sequence of tokens including x, y, eq, etc.
        
        try:
            chunk = json.loads(chunk) # convert the sequence tokens to a dictionary
        except:
            print("Couldn't convert to json: {}".format(chunk))
            
        # find the number of variables in the equation
        eq = chunk[self.target]
        vars = re.finditer('x[\d]+',eq) 
        numVars = 0
        for v in vars:
            v = v.group(0).strip('x')
            v = eval(v)
            v = int(v)
            if v > numVars:
                numVars = v
        
        # encode every character in the equation to an integer
        # < is SOS, > is EOS
        if self.addVars:
            dix = [self.stoi[s] for s in '<'+str(numVars)+':'+eq+'>']
        else:
            dix = [self.stoi[s] for s in '<'+eq+'>']
        inputs = dix[:-1]
        outputs = dix[1:]
        
        # add the padding to the equations
        paddingSize = max(self.block_size-len(inputs),0)
        paddingList = [self.paddingID]*paddingSize
        inputs += paddingList
        outputs += paddingList
        
        # make sure it is not more than what should be
        inputs = inputs[:self.block_size]
        outputs = outputs[:self.block_size]
        
        # extract points from the input sequence
        # maxX = max(chunk['X'])
        # maxY = max(chunk['Y'])
        # minX = min(chunk['X'])
        # minY = min(chunk['Y'])
        eps = 1e-5
        points = torch.zeros(self.numVars+self.numYs, self.numPoints)
        for idx, xy in enumerate(zip(chunk['X'], chunk['Y'])):
            
            x = xy[0]
            #x = [(e-minX[eID])/(maxX[eID]-minX[eID]+eps) for eID, e in enumerate(x)] # normalize x
            x = x + [0]*(max(self.numVars-len(x),0)) # padding

            y = [xy[1]] if type(xy[1])== float else xy[1]
            #y = [(e-minY)/(maxY-minY+eps) for e in y]
            y = y + [0]*(max(self.numYs-len(y),0)) # padding
            p = x+y # because it is only one point 
            p = torch.tensor(p)
            #replace nan and inf
            p = torch.nan_to_num(p, nan=self.threshold[1], 
                                 posinf=self.threshold[1], 
                                 neginf=self.threshold[0])
            # p[p>self.threshold[1]] = self.threshold[1] # clip the upper bound
            # p[p<self.threshold[0]] = self.threshold[0] # clip the lower bound
            points[:,idx] = p

        # Normalize points between zero and one # DxN
        # minP = points.min(dim=1, keepdim=True)[0]
        # maxP = points.max(dim=1, keepdim=True)[0]
        # points -= minP
        # points /= (maxP-minP+eps)
        points -= points.mean()
        points /= points.std()
        points = torch.nan_to_num(points, nan=self.threshold[1],
                                 posinf=self.threshold[1],
                                 neginf=self.threshold[0])
        #points += torch.normal(0, 0.05, size=points.shape) # add a guassian noise
        
        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        numVars = torch.tensor(numVars, dtype=torch.long)
        return inputs, outputs, points, numVars

def processDataFiles(files):
    text = ''""
    for f in tqdm(files):
        with open(f, 'r') as h: 
            lines = h.read() # don't worry we won't run out of file handles
            if lines[-1]==-1:
                lines = lines[:-1]
            text += lines #json.loads(line)        
    return text

def lossFunc(constants, eq, X, Y, eps=1e-5):
    err = 0
    eq = eq.replace('C','{}').format(*constants)

    for x,y in zip(X,Y):
        eqTemp = eq + ''
        if type(x) == np.float32:
            x = [x]
        for i,e in enumerate(x):
            # make sure e is not a tensor
            if type(e) == torch.Tensor:
                e = e.item()
            eqTemp = eqTemp.replace('x{}'.format(i+1), str(e))
        try:
            yHat = eval(eqTemp)
        except:
            print('Exception has been occured! EQ: {}, OR: {}'.format(eqTemp, eq))
            continue
            yHat = 100
        try:
            # handle overflow
            err += relativeErr(y, yHat) #(y-yHat)**2
        except:
            print('Exception has been occured! EQ: {}, OR: {}, y:{}-yHat:{}'.format(eqTemp, eq, y, yHat))
            continue
            err += 10
        
    err /= len(Y)
    return err

def generateDataStrEq(eq, n_points=2, n_vars=3,
                        decimals=4, supportPoints=None, min_x=0, max_x=3):
    X = []
    Y= []
    # TODO: Need to make this faster
    for p in range(n_points):
        if supportPoints is None:
            if type(min_x) == list:
                x = []
                for _ in range(n_vars):
                    idx = np.random.randint(len(min_x))
                    x += list(np.round(np.random.uniform(min_x[idx], max_x[idx], 1), decimals))
            else:
                x = list(np.round(np.random.uniform(min_x, max_x, n_vars), decimals))
        else:
            x = supportPoints[p]

        tmpEq = eq + ''
        for nVID in range(n_vars):
            tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))
        y = float(np.round(eval(tmpEq), decimals))
        X.append(x)
        Y.append(y)
    return X, Y