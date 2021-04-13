#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from GenerateData import generate_random_eqn_raw, eqn_to_str, create_dataset_from_raw_eqn, simplify_formula, dataGen

def processData(numSamples, nv, decimals, 
                template, dataPath, fileID, time, 
                supportPoints=None, numberofPoints=30,
                xRange=[0.1,3.1], testPoints=False,
                testRange=[0.0,6.0], n_levels = 3,
                allow_constants=True, 
                const_range=[-0.4, 0.4],
                const_ratio=0.8,
                op_list=[
                    "id", "add", "mul", "div", 
                    "sqrt", "sin", "exp", "log"]
                ):
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            x,y,cleanEqn, skeletonEqn, xT, yT = dataGen( 
                                                nv = nv, decimals = decimals, 
                                                numberofPoints=numberofPoints, 
                                                supportPoints=supportPoints,
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
        
        # hold data in the structure
        structure['X'] = list(x)
        structure['Y'] = y
        structure['EQ'] = cleanEqn
        structure['Skeleton'] = skeletonEqn
        structure['XT'] = list(xT)
        structure['YT'] = yT

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
    seed = 2021
    np.random.seed(seed=seed) # fix the seed for reproducibility

    numVars = [1] #list(range(31)) #[1,2,3,4,5]
    decimals = 2
    numberofPoints = [1,30] # only usable if support points has not been provided
    numSamples = 1000 # number of generated samples
    folder = './Dataset'
    dataPath = folder +'/{}_{}_{}.json'
    supportPoints = np.linspace(0.1,3.1,30)
    supportPoints = [[np.round(p,decimals)] for p in supportPoints]
    #supportPoints = None # uncomment this line if you don't want to use support points
    xRange = [0.0,3.0]
    testPoints = True
    testRange = [3.1,6.0]
    n_levels = 2
    allow_constants = True
    const_range = [-1, 1]
    const_ratio = 0.5
    op_list=[
                "id", "add", "mul",
                "sqrt", "sin", 
            ]

    print(os.mkdir(folder) if not os.path.isdir(folder) else 'We do have the path already!')

    template = {'X':[], 'Y':0.0, 'EQ':''}
    fileID = 0
    #mp.set_start_method('spawn')
    #q = mp.Queue()
    processes = []
    for i, nv in enumerate(numVars):
        from datetime import datetime
        now = datetime.now()
        time = '{}_'.format(i) + now.strftime("%d%m%Y_%H%M%S")
        print('Processing equations with {} variables!'.format(nv))
        #p = mp.Process(target=processData, args=(numSamples, nv, decimals, template, dataPath, fileID, time, supportPoints, numberofPoints,seed,))

        p = mp.Process(target=processData, 
                       args=(
                                numSamples, nv, decimals, template, 
                                dataPath, fileID, time, supportPoints, numberofPoints,
                                xRange, testPoints, testRange, n_levels, 
                                allow_constants, const_range,
                                const_ratio, op_list
                            )
                       )

        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()

