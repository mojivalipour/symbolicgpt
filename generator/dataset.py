#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
from GenerateData import generate_random_eqn_raw, eqn_to_str, create_dataset_from_raw_eqn, simplify_formula, dataGen

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
                ):
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            x,y,cleanEqn,skeletonEqn = dataGen( 
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
                                                const_ratio=const_ratio
                                               )
        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i-1
            continue

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
    # Config
    seed = 2021 # 2021 Train, 2022 Val, 2023 Test
    #from GenerateData import seed
    import random
    random.seed(seed)
    np.random.seed(seed=seed) # fix the seed for reproducibility

    #NOTE: For linux you can only use unique numVars, in Windows, it is possible to use [1,2,3,4] * 10!
    numVars = [1] #list(range(31)) #[1,2,3,4,5]
    decimals = 4
    numberofPoints = [20,21] # only usable if support points has not been provided
    numSamples = 10000 # number of generated samples
    folder = './Dataset'
    dataPath = folder +'/{}_{}_{}.json'

    testPoints = False
    xRange = [-1.0,4.0]
    testRange = [4.1,8.0]

    supportPoints = None
    #supportPoints = np.linspace(xRange[0],xRange[1],numberofPoints[1])
    #supportPoints = [[np.round(p,decimals)] for p in supportPoints]
    #supportPoints = [[np.round(p,decimals), np.round(p,decimals)] for p in supportPoints]
    #supportPoints = [[np.round(p,decimals) for i in range(numVars[0])] for p in supportPoints]

    supportPointsTest = None
    #supportPoints = None # uncomment this line if you don't want to use support points
    #supportPointsTest = np.linspace(xRange[0],xRange[1],numberofPoints[1])
    #supportPointsTest = [[np.round(p,decimals) for i in range(numVars[0])] for p in supportPointsTest]
    
    n_levels = 3
    allow_constants = True
    const_range = [-1, 1]
    const_ratio = 0.0
    op_list=[
                "id", "add", "sub", "mul", "div",
                "sin", "exp", "log" #, "pow", "cos"
            ]

    sortY = False # if the data is sorted based on y

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
                                xRange, testPoints, testRange, n_levels, 
                                allow_constants, const_range,
                                const_ratio, op_list, sortY
                            )
                       )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()


