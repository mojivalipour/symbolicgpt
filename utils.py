import re
import json
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import minimize
from torch.utils.data import Dataset
from torch.nn import functional as F
from numpy import * # to override the math functions
from matplotlib import pyplot as plt

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

# @torch.no_grad()
# def sample_from_model(model, x, steps, points=None, variables=None, temperature=1.0, sample=False, top_k=None):
#     """
#     take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
#     the sequence, feeding the predictions back into the model each time. Clearly the sampling
#     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
#     of block_size, unlike an RNN that has an infinite context window.
#     """
#     block_size = model.get_block_size()
#     model.eval()
#     for k in range(steps):
#         x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
#         logits, _ = model(x_cond, points=points, variables=variables)
#         # pluck the logits at the final step and scale by temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
#         # apply softmax to convert to probabilities
#         probs = F.softmax(logits, dim=-1)
#         # sample from the distribution or take the most likely
#         if sample:
#             ix = torch.multinomial(probs, num_samples=1)
#         else:
#             _, ix = torch.topk(probs, k=1, dim=-1)
#         # append to the sequence and continue
#         x = torch.cat((x, ix), dim=1)

#     return x

#use nucleus sampling from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    #TODO: support for batch size more than 1
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def sample_from_model(model, x, steps, points=None, variables=None, temperature=1.0, sample=False, top_k=0.0, top_p=0.0):
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
        logits = logits[0, -1, :] / temperature
        # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix.unsqueeze(0)), dim=1)

    return x

def plot_and_save_results(resultDict, fName, pconf, titleTemplate, textTest, modelKey='SymbolicGPT'):
    if isinstance(resultDict, dict):
        # plot the error frequency for model comparison
        num_eqns = len(resultDict[fName][modelKey]['err'])
        num_vars = pconf.numberofVars
        title = titleTemplate.format(num_eqns, num_vars)

        models = list(key for key in resultDict[fName].keys() if len(resultDict[fName][key]['err'])==num_eqns)
        lists_of_error_scores = [resultDict[fName][key]['err'] for key in models if len(resultDict[fName][key]['err'])==num_eqns]
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

        with open(fName, 'w', encoding="utf-8") as o:
            for i in range(num_eqns):
                err = resultDict[fName][modelKey]['err'][i]
                eq = resultDict[fName][modelKey]['trg'][i]
                predicted = resultDict[fName][modelKey]['prd'][i]
                print('Test Case {}.'.format(i))
                print('Target:{}\nSkeleton:{}'.format(eq, predicted))
                print('Err:{}'.format(err))
                print('') # just an empty line

            
                o.write('Test Case {}/{}.\n'.format(i,len(textTest)-1))

                o.write('{}\n'.format(eq))
                o.write('{}:\n'.format(modelKey))
                o.write('{}\n'.format(predicted))

                o.write('{}\n{}\n\n'.format( 
                                        predicted,
                                        err
                                        ))

                print('Avg Err:{}'.format(np.mean(resultDict[fName][modelKey]['err'])))

def tokenize_predict_and_evaluate(i, inputs, points, outputs, variables, 
                                  train_dataset, textTest, trainer, model, resultDict,
                                  numTests, variableEmbedding, blockSize, fName,
                                  modelKey='SymbolicGPT', device='cpu'):
    
    eq = ''.join([train_dataset.itos[int(i)] for i in outputs[0]])
    eq = eq.strip(train_dataset.paddingToken).split('>')
    eq = eq[0] #if len(eq[0])>=1 else eq[1]
    eq = eq.strip('<').strip(">")
    print(eq)
    if variableEmbedding == 'STR_VAR':
            eq = eq.split(':')[-1]

    t = json.loads(textTest[i])

    inputs = inputs[:,0:1].to(device)
    points = points.to(device)
    # points = points[:,:numPoints] # filter anything more than maximum number of points
    variables = variables.to(device)

    bestErr = 10000000
    bestPredicted = 'C'
    for i in range(numTests):
        
        predicted, err = generate_sample_and_evaluate(
                            model, t, eq, inputs, 
                            blockSize, points, variables, 
                            train_dataset, variableEmbedding)

        if err < bestErr:
            bestErr = err
            bestPredicted = predicted
    
    resultDict[fName][modelKey]['err'].append(bestErr)
    resultDict[fName][modelKey]['trg'].append(eq)
    resultDict[fName][modelKey]['prd'].append(bestPredicted)

    return eq, bestPredicted, bestErr

def generate_sample_and_evaluate(model, t, eq, inputs, 
                                 blockSize, points, variables, 
                                 train_dataset, variableEmbedding):

    
    outputsHat = sample_from_model(model, 
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
        predicted = predicted.split(':')[-1]

    predicted = predicted.strip(train_dataset.paddingToken).split('>')
    predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
    predicted = predicted.strip('<').strip(">")
    predicted = predicted.replace('Ce','C*e')

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
    
    print('\nTarget:{}'.format(eq))
    print('Skeleton+LS:{}'.format(predicted))
    print('Err:{}'.format(err))
    print('-'*10)

    if type(err) is np.complex128 or np.complex:
        err = abs(err.real)

    return predicted, err

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
                 numVars, numYs, numPoints, target='EQ', 
                 addVars=False, const_range=[-0.4, 0.4],
                 xRange=[-3.0,3.0], decimals=4, augment=False):

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

        self.const_range = const_range
        self.xRange = xRange
        self.decimals = decimals
        self.augment = augment
    
    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        # grab an example from the data
        chunk = self.data[idx] # sequence of tokens including x, y, eq, etc.
        
        try:
            chunk = json.loads(chunk) # convert the sequence tokens to a dictionary
        except Exception as e:
            print("Couldn't convert to json: {} \n error is: {}".format(chunk, e))
            # try the previous example
            idx = idx - 1 
            idx = idx if idx>=0 else 0
            chunk = self.data[idx]
            chunk = json.loads(chunk) # convert the sequence tokens to a dictionary
            
        # find the number of variables in the equation
        printInfoCondition = random.random() < 0.0000001
        eq = chunk[self.target]
        if printInfoCondition:
            print(f'\nEquation: {eq}')
        vars = re.finditer('x[\d]+',eq) 
        numVars = 0
        for v in vars:
            v = v.group(0).strip('x')
            v = eval(v)
            v = int(v)
            if v > numVars:
                numVars = v

        if self.target == 'Skeleton' and self.augment:
            threshold = 5000
            # randomly generate the constants
            cleanEqn = ''
            for chr in eq:
                if chr == 'C':
                    # genereate a new random number
                    chr = '{}'.format(np.random.uniform(self.const_range[0], self.const_range[1]))
                cleanEqn += chr

            # update the points
            nPoints = np.random.randint(*self.numPoints) #if supportPoints is None else len(supportPoints)
            try:
                if printInfoCondition:
                    print('Org:',chunk['X'], chunk['Y'])

                X, y = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=self.numVars,
                                         decimals=self.decimals, min_x=self.xRange[0], 
                                         max_x=self.xRange[1])

                # replace out of threshold with maximum numbers
                y = [e if abs(e)<threshold else np.sign(e) * threshold for e in y]

                # check if there is nan/inf/very large numbers in the y
                conditions = (np.isnan(y).any() or np.isinf(y).any()) or len(y) == 0 or (abs(min(y)) > threshold or abs(max(y)) > threshold)
                if not conditions:
                    chunk['X'], chunk['Y'] = X, y

                if printInfoCondition:
                    print('Evd:',chunk['X'], chunk['Y'])
            except Exception as e: 
                # for different reason this might happend including but not limited to division by zero
                print("".join([
                    f"We just used the original equation and support points because of {e}. ",
                    f"The equation is {eq}, and we update the equation to {cleanEqn}",
                ]))
 
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
        points = torch.zeros(self.numVars+self.numYs, self.numPoints[1]-1)
        for idx, xy in enumerate(zip(chunk['X'], chunk['Y'])):

            # don't let to exceed the maximum number of points
            if idx >= self.numPoints[1]-1:
                break
            
            x = xy[0]
            #x = [(e-minX[eID])/(maxX[eID]-minX[eID]+eps) for eID, e in enumerate(x)] # normalize x
            x = x + [0]*(max(self.numVars-len(x),0)) # padding

            y = [xy[1]] if type(xy[1])==float or type(xy[1])==np.float64 else xy[1]

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
        # if printInfoCondition:
        #     print(f'Points: {points}')

        # points -= points.mean()
        # points /= points.std()
        points = torch.nan_to_num(points, nan=self.threshold[1],
                                 posinf=self.threshold[1],
                                 neginf=self.threshold[0])

        # if printInfoCondition:
        #     print(f'Points: {points}')
        #points += torch.normal(0, 0.05, size=points.shape) # add a guassian noise
        
        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        numVars = torch.tensor(numVars, dtype=torch.long)
        return inputs, outputs, points, numVars

def processDataFiles(files):
    text = ""
    for f in tqdm(files):
        with open(f, 'r') as h: 
            lines = h.read() # don't worry we won't run out of file handles
            if lines[-1]==-1:
                lines = lines[:-1]
            #text += lines #json.loads(line)    
            text = ''.join([lines,text])    
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
                      decimals=4, supportPoints=None, 
                      min_x=0, max_x=3):
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
            assert len(x)!=0, "For some reason, we didn't generate the points correctly!"
        else:
            x = supportPoints[p]

        tmpEq = eq + ''
        for nVID in range(n_vars):
            tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))
        y = float(np.round(eval(tmpEq), decimals))
        X.append(x)
        Y.append(y)
    return X, Y