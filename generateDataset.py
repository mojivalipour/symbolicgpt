#!/usr/bin/env python
# coding: utf-8

import os
import json
import ProGED
import random
import numpy as np
from numpy import *
from tqdm import tqdm
from datetime import datetime
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.generate import generate_models

# GRAMMAR_LIBRARY = {
#     "universal": construct_grammar_universal,
#     "universal-dim": construct_grammar_universal_dim,
#     "rational": construct_grammar_rational,
#     "simplerational": construct_grammar_simplerational,
#     "polytrig": construct_grammar_polytrig,
#     "trigonometric": construct_grammar_trigonometric,
#     "polynomial": construct_grammar_polynomial}

# Config
numSamples = 1000
numVars = 2
seed = 2022 # 2021: Train, 2022: Val, 2023: Test
numPoints = [20,21]
decimals = 4
trainRange = [-1.0,1.0]
testRange = [1.1,4.0]
constantsRange = [-1,1]
grammerType = 'universal' # universal/polynomial
template = {'EQ':'', 'Skeleton':'', 'X':[], 'Y':0.0, 'XT':[], 'YT':0.0,}
folder = './Dataset'

os.makedirs(folder, exist_ok=True)
now = datetime.now()
time = now.strftime("%d%m%Y_%H%M%S")
dataPath = folder +'/id{}_nv{}_np{}_trR{}_teR{}_t{}.json'.format('{}', numVars, numPoints, 
                                                                 trainRange,
                                                                 testRange, 
                                                                 time)
print(dataPath)

# add a safe wrapper for numpy math functions
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

# Mean square error
def relativeErr(y, y_hat):
    y_hat = np.reshape(y_hat, [1, -1])[0]
    y_gold = np.reshape(y, [1, -1])[0]
    our_sum = 0
    for i in range(len(y_gold)):
        if y_gold[i] < 1: 
            # use regular MSE
            our_sum += (y_hat[i] - y_gold[i]) ** 2
        else:
            # use relative MSE
            our_sum += ((y_hat[i] - y_gold[i])/y_gold[i]) ** 2

    return our_sum / len(y_gold)

# fixed the seed
np.random.seed(seed)
random.seed(seed)
rng = np.random.RandomState(seed)

symbols = {"x":['x{}'.format(i+1) for i in range(numVars)], "start":"S", "const":"C"}

if grammerType == 'polynomial':
    grammer = grammar_from_template("polynomial",
                {"variables":["'x1'","'x2'"],
                "p_vars":[0.5, 0.5],
                "functions":["'1.0*'"],
                "p_S":[0.4, 0.6], 
                "p_T":[0.4, 0.6],
                "p_R":[0.6, 0.4],
                "p_F":[0.5],
                })  
else:
    grammer = grammar_from_template("universal", 
        {"functions":["sin", "sqrt", "exp", "log"], 
        "variables":["'x{}'".format(i+1) for i in range(numVars)],
        "p_sum":[0.2, 0.2, 0.6], 
        "p_mul": [0.3, 0.1, 0.6], 
        "p_rec": [0.2, 0.4, 0.4], 
        "p_vars":[1/numVars for i in range(numVars)],
        "p_functs":[0.7, 0.1, 0.1, 0.05, 0.05]})

# Generate Equations
equations = []
targetSamples = numSamples + 0
while len(equations) < numSamples:
    models = generate_models(grammer, symbols, strategy_settings = {"N":targetSamples}) # the output is an ModelBox object
    equations.extend([eq for eq in models])
    targetSamples = abs(numSamples - len(equations))

# Generate the data
fileID = 1

for eq in tqdm(equations):
    skeletonEqn = eq.__str__().replace('E', '0.0').replace('I', '0.0') # convert the object to string
    
    chosenPoints = np.random.randint(numPoints[0],numPoints[1]) # for each equation choose the number of points randomly
    
    # find all constants in the generated equation, generate a random number based on the given boundry
    constants = [random.uniform(constantsRange[0], constantsRange[1]) for i,x in enumerate(skeletonEqn) if x=='C']            
    eq = skeletonEqn.replace('C','{}').format(*constants) if len(constants)>0 else skeletonEqn

    saveEq = True
    
    # for each variable, generate the same number of points (x: (numPoints, numVars))
    X = np.round(rng.uniform(low=trainRange[0], high=trainRange[1], size=(chosenPoints,numVars)), decimals) # generate random points uniformly
    
    # calculate y based on x
    Y = []
    for point in X:
        tmpEq = eq + '' # copy the string
        for varId in range(numVars):
            tmpEq = tmpEq.replace('x{}'.format(varId+1),str(np.round(point[varId], decimals)))
        try: 
            y = eval(tmpEq)
            if type(y) is np.complex128 or type(y) is np.complex:
                print('Type was complex! Why?: {}'.format(tmpEq))
                y = 0 #abs(err.real)
                saveEq = False
        except ZeroDivisionError:
            print('Zero Division: {}'.format(tmpEq))
            y = 0
            saveEq = False
        except OverflowError:
            print('Overflow Error: {}'.format(tmpEq))
            y = 0
            saveEq = False
        except:
            saveEq = False
            raise Exception('Err to process this equation: {}, original:{}'.format(tmpEq, skeletonEqn)) 

        Y.append(round(y, decimals))
        
    # generate xt for the test range
    XT = np.round(rng.uniform(low=testRange[0], high=testRange[1], size=(chosenPoints,numVars)), decimals) # generate random points uniformly
    
    # calculate yt based on xt
    YT = []
    for point in XT:
        tmpEq = eq + '' # copy the string
        for varId in range(numVars):
            tmpEq = tmpEq.replace('x{}'.format(varId+1),str(point[varId]))
        try: 
            y = eval(tmpEq)
            if type(y) is np.complex128 or type(y) is np.complex:
                print('Type was complex! Why?: {}'.format(tmpEq))
                y = 0 #abs(err.real)
                saveEq = False
        except ZeroDivisionError:
            print('Zero Division: {}'.format(tmpEq))
            y = 0
            saveEq = False
        except OverflowError:
            print('Overflow Error: {}'.format(tmpEq))
            y = 0
            saveEq = False
        except:
            saveEq = False
            raise Exception('Err to process this equation: {}, original:{}'.format(tmpEq, skeletonEqn)) 
        YT.append(round(y, decimals))
    
    if not saveEq:
        # ignore this sample, generate another one
        i = i-1 
        continue
    
    structure = template.copy() # copy the template
    
    # hold data in the structure
    structure['X'] = X.tolist()
    structure['Y'] = Y
    structure['Skeleton'] = skeletonEqn
    structure['EQ'] = eq
    structure['XT'] = XT.tolist()
    structure['YT'] = YT
    
    # write to a file
    outputPath = dataPath.format(fileID)
    if os.path.exists(outputPath):
        fileSize = os.path.getsize(outputPath)
        if fileSize > 500000000: # 500 MB
            fileID +=1 
        
    with open(outputPath, "a", encoding="utf-8") as h:
        json.dump(structure, h, ensure_ascii=False)
        h.write('\n')