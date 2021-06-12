#!/usr/bin/env python
# coding: utf-8

import re
import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
from generator.treeBased.generateData import dataGen
from utils import * # TODO: replace with a safer import

def processData(numSamples, nv, decimals, 
                template, dataPath, fileID, time, 
                supportPoints=None, 
                supportPointsTest=None,
                numberofPoints=30,
                xRange=[0.1,3.1], testPoints=False,
                testRange=[0.0,6.0], n_levels = 3,
                allow_constants=True, 
                const_range=[-0.4, 0.4],
                const_ratio=0.8,
                op_list=[
                    "id", "add", "mul", "div", 
                    "sqrt", "sin", "exp", "log"],
                sortY=False,
                exponents= [3,4,5,6],
                numSamplesEachEq=1,
                threshold = 100,
                ):
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            cleanEqn,skeletonEqn, currEqn = dataGen( 
                                                    nv = nv, decimals = decimals, 
                                                    numberofPoints=numberofPoints, 
                                                    supportPoints=supportPoints,
                                                    supportPointsTest=supportPointsTest,
                                                    xRange=xRange,
                                                    testPoints=testPoints,
                                                    testRange=testRange,
                                                    n_levels=n_levels,
                                                    op_list=op_list,
                                                    allow_constants=allow_constants, 
                                                    const_range=const_range,
                                                    const_ratio=const_ratio,
                                                    exponents=exponents
                                                )
        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i-1
            continue

        # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
        exps = re.findall(r"(\*\*[0-9\.]+)", skeletonEqn)
        for ex in exps:
            # correct the exponent
            cexp = '**'+str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2,exponents[-1]+1))
            # replace the exponent
            skeletonEqn = skeletonEqn.replace(ex, cexp)     

        for e in range(numSamplesEachEq):
            # replace the constants with new ones
            cleanEqn = ''
            for chr in skeletonEqn:
                if chr == 'C':
                    # genereate a new random number
                    chr = '{}'.format(np.random.uniform(const_range[0], const_range[1]))
                cleanEqn += chr

            if 'I' in cleanEqn or 'zoo' in cleanEqn:
                # repeat the equation generation
                print('This equation has been rejected: {}'.format(cleanEqn))
                i -= 1
                break

            # generate new data points
            nPoints = np.random.randint(
                    *numberofPoints) if supportPoints is None else len(supportPoints)

            data = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=nv,
                                        decimals=decimals, supportPoints=supportPoints, min_x=xRange[0], max_x=xRange[1])
            # if testPoints:
            #     dataTest = generateDataStrEq(currEqn, n_points=numberofPoints, n_vars=nv, decimals=decimals,
            #                                         supportPoints=supportPointsTest, min_x=testRange[0], max_x=testRange[1]))   

            # use the new x and y
            x,y = data

            # check if there is nan/inf/very large numbers in the y
            if np.isnan(y).any() or np.isinf(y).any() or np.any([abs(e)>threshold for e in y]):
                # repeat the equation generation
                i -= 1
                break

            # just make sure there is no samples out of the threshold
            if abs(min(y)) > threshold or abs(max(y)) > threshold:
                raise 'Err: Min:{},Max:{},Threshold:{}, \n Y:{} \n Eq:{}'.format(min(y), max(y), threshold, y, cleanEqn)

            # sort data based on Y
            if sortY:
                x,y = zip(*sorted(zip(x,y), key=lambda d: d[1]))
            
            # hold data in the structure
            structure['X'] = list(x)
            structure['Y'] = y
            structure['Skeleton'] = skeletonEqn
            structure['EQ'] = cleanEqn

            outputPath = dataPath.format(fileID, nv, time)
            if os.path.exists(outputPath):
                fileSize = os.path.getsize(outputPath)
                if fileSize > 500000000: # 500 MB
                    fileID +=1 
            with open(outputPath, "a", encoding="utf-8") as h:
                json.dump(structure, h, ensure_ascii=False)
                h.write('\n')

def main():
    # Config
    seed = 2021 # 2021 Train, 2022 Val, 2023 Test, you have to change the generateData.py seed as well
    #from GenerateData import seed
    import random
    random.seed(seed)
    np.random.seed(seed=seed) # fix the seed for reproducibility

    #NOTE: For linux you can only use unique numVars, in Windows, it is possible to use [1,2,3,4] * 10!
    numVars = [1] #list(range(31)) #[1,2,3,4,5]
    decimals = 8
    numberofPoints = [30,31] # only usable if support points has not been provided
    numSamples = 10000 # number of generated samples
    folder = './Dataset'
    dataPath = folder +'/{}_{}_{}.json'

    testPoints = False
    trainRange = [-3.0,3.0] 
    testRange = [[-5.0, 3.0],[-3.0, 5.0]] # this means Union((-5,-1),(1,5))

    supportPoints = None
    #supportPoints = np.linspace(xRange[0],xRange[1],numberofPoints[1])
    #supportPoints = [[np.round(p,decimals)] for p in supportPoints]
    #supportPoints = [[np.round(p,decimals), np.round(p,decimals)] for p in supportPoints]
    #supportPoints = [[np.round(p,decimals) for i in range(numVars[0])] for p in supportPoints]

    supportPointsTest = None
    #supportPoints = None # uncomment this line if you don't want to use support points
    #supportPointsTest = np.linspace(xRange[0],xRange[1],numberofPoints[1])
    #supportPointsTest = [[np.round(p,decimals) for i in range(numVars[0])] for p in supportPointsTest]
    
    n_levels = 4
    allow_constants = True
    const_range = [-2.1, 2.1]
    const_ratio = 0.5
    op_list=[
                "id", "add", "mul",
                "sin", "pow", "cos", 
                "exp", "div", "sub", "log"
            ]
    exponents=[3, 4, 5, 6]

    sortY = False # if the data is sorted based on y
    numSamplesEachEq = 50

    print(os.mkdir(folder) if not os.path.isdir(folder) else 'We do have the path already!')

    template = {'X':[], 'Y':0.0, 'EQ':''}
    fileID = 0
    #mp.set_start_method('spawn')
    #q = mp.Queue()
    processes = []
    for i, nv in enumerate(numVars):
        now = datetime.now()
        time = '{}_'.format(i) + now.strftime("%d%m%Y_%H%M%S")
        print('Processing equations with {} variables!'.format(nv))

        p = mp.Process(target=processData, 
                       args=(
                                numSamples, nv, decimals, template, 
                                dataPath, fileID, time, supportPoints, 
                                supportPointsTest,
                                numberofPoints,
                                trainRange, testPoints, testRange, n_levels, 
                                allow_constants, const_range,
                                const_ratio, op_list, sortY, exponents,
                                numSamplesEachEq
                            )
                       )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()


