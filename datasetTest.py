#!/usr/bin/env python
# coding: utf-8

import re
import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from generator.treeBased.generateData import dataGen
from utils import * # TODO: replace with a safer import

def processData(numSamples, nv, decimals,
                template, dataPath, fileID, time,
                supportPoints=None,
                supportPointsTest=None,
                numberofPoints=[20,250],
                xRange=[0.1, 3.1], testPoints=False,
                testRange=[0.0, 6.0], n_levels=3,
                allow_constants=True,
                const_range=[-0.4, 0.4],
                const_ratio=0.8,
                op_list=[
                    "id", "add", "mul", "div",
                    "sqrt", "sin", "exp", "log"],
                sortY=False,
                exponents=[3,4,5,6],
                threshold=1000,
                templatesEQs=None,
                templateProb=0.4,
                ):
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            _, skeletonEqn, _ = dataGen( 
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
            if templatesEQs != None and np.random.rand() < templateProb: 
                # by a chance, replace the skeletonEqn with a given templates
                idx = np.random.randint(len(templatesEQs[nv]))
                skeletonEqn = templatesEQs[nv][idx]

        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i - 1
            continue

        # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
        exps = re.findall(r"(\*\*[0-9\.]+)", skeletonEqn)
        for ex in exps:
            # correct the exponent
            cexp = '**'+str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2,exponents[-1]+1))
            # replace the exponent
            skeletonEqn = skeletonEqn.replace(ex, cexp)     

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
            continue

        # create a set of points
        nPoints = np.random.randint(
                *numberofPoints) if supportPoints is None else len(supportPoints)

        data = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=nv,
                                    decimals=decimals, supportPoints=supportPoints, min_x=xRange[0], max_x=xRange[1])
        # use the new x and y
        x,y = data

        if testPoints:
            dataTest = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=nv, decimals=decimals,
                                                supportPoints=supportPointsTest, min_x=testRange[0], max_x=testRange[1])   
            xT, yT = dataTest

        # check if there is nan/inf/very large numbers in the y
        if np.isnan(y).any() or np.isinf(y).any() or np.any([abs(e)>threshold for e in y]):
            # repeat the equation generation
            i -= 1
            print('{} has been rejected because of wrong value in y.'.format(skeletonEqn))
            continue

        if len(y) == 0: # if for whatever reason the y is empty
            print('Empty y, x: {}, most of the time this is because of wrong numberofPoints: {}'.format(x, numberofPoints))
            e -= 1
            continue

        # just make sure there is no samples out of the threshold
        if abs(min(y)) > threshold or abs(max(y)) > threshold:
            raise 'Err: Min:{},Max:{},Threshold:{}, \n Y:{} \n Eq:{}'.format(min(y), max(y), threshold, y, cleanEqn)

        # sort data based on Y
        if sortY:
            x,y = zip(*sorted(zip(x,y), key=lambda d: d[1]))

        # hold data in the structure
        structure['X'] = list(x)
        structure['Y'] = y
        structure['EQ'] = cleanEqn
        structure['Skeleton'] = skeletonEqn
        structure['XT'] = list(xT)
        structure['YT'] = yT

        print('\n EQ: {}'.format(skeletonEqn))

        outputPath = dataPath.format(fileID, nv, time)
        if os.path.exists(outputPath):
            fileSize = os.path.getsize(outputPath)
            if fileSize > 500000000:  # 500 MB
                fileID += 1
        with open(outputPath, "a", encoding="utf-8") as h:
            json.dump(structure, h, ensure_ascii=False)
            h.write('\n')

def main():
    # Config
    seed = 2023 # 2021 Train, 2022 Val, 2023 Test, you have to change the generateData.py seed as well
    #from GenerateData import seed
    import random
    random.seed(seed)
    np.random.seed(seed=seed) # fix the seed for reproducibility

    #NOTE: For linux you can only use unique numVars, in Windows, it is possible to use [1,2,3,4] * 10!
    numVars = list(range(1,10)) #list(range(31)) #[1,2,3,4,5]
    decimals = 4
    numberofPoints = [20,250] # only usable if support points has not been provided
    numSamples = 2500 // len(numVars) # number of generated samples
    folder = './Dataset'
    dataPath = folder +'/{}_{}_{}.json'

    testPoints = True
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
                "sin", "pow", "cos", "sqrt",
                "exp", "div", "sub", "log",
                "arcsin",
            ]
    exponents=[3, 4, 5, 6]
    
    sortY = False # if the data is sorted based on y
    threshold = 5000
    templateProb = 1.0 #0.3 # the probability of generating an equation from the templates
    templatesEQs = {
        1: [
            # NGUYEN
            'C*x1**3+C*x1**2+C*x1+C', 
            'C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            'C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            'C*x1**6+C*x1**5+C*x1**4+C*x1**3+C*x1**2+C*x1+C',
            'C*sin(C*x1**2)*cos(C*x1+C)+C',
            'C*sin(C*x1+C)+C*sin(C*x1+C*x1**2)+C',
            'C*log(C*x1+C)+C*log(C*x1**2+C)+C',
            'C*sqrt(C*x1+C)+C',
            ],
        2: [
            # NGUYEN
            'C*sin(C*x1+C)+C*sin(C*x2**2+C)+C',
            'C*sin(C*x1+C)*cos(C*x2+C)+C',
            'C*x1**x2+C',
            'C*x1**4+C*x1**3+C*x2**2+C*x2+C',
            # AI Faynman
            'C*exp(C*x1**2+C)/sqrt(C*x2+C)+C',
            'C*x1*x2+C',
            'C*1/2*x1*x2**2+C',
            'C*x1/x2+C',
            'C*arcsin(C*x1*sin(C*x2+C)+C)+C',
            'C*(C*x1/(2*pi)+C)*x2+C',
            'C*3/2*x1*x2+C',
            'C*x1/(C*4*pi*x2**2+C)+C',
            'C*x1*x2**2/2+C',
            'C*1+C*x1*x2/(C*1-C*(C*x1*x2/3+C)+C)+C',
            'C*x1*x2**2+C',
            'C*x1/(2*(1+C*x2+C))+C',
            'C*x1*(C*x2/(2*pi)+C)+C',
            ], 
        3: [
            # AI Faynman
            'C*exp(C*(x1/x2)**2)/(C*sqrt(2*x3)*x2+C)+C',
            'C*x1/sqrt(1-x2**2/x3**2+C)+C',
            'C*x1*x2*x3+C',
            'C*x1*x2/sqrt(C*1-C*x2**2/x3**2+C)+C',
            'C*(C*x1+C*x2+C)/(C*1+C*x1*x2/x3**2+C)+C',
            'C*x1*x3*sin(C*x2+C)+C',
            'C*1/(C*1/x1+C*x2/x3+C)+C',
            'C*x1*sin(C*x2*x3/2+C)**2/sin(x3/2)**2+C',
            'C*arcsin(C*x1/(C*x2*x3+C)+C)+C',
            'C*x1/(C*1-C*x2/x3+C)+C',
            'C*(1+C*x1/x3+C)/sqrt(1-C*x1**2/x3**2+C)*x2+C',
            'C*(C*x1/(C*x3+C)+C)*x2+C',
            'C*x1+C*x2+C*2*sqrt(x1*x2)*cos(x3)+C',
            'C*1/(x1-1)*x2*x3+C',
            'C*x1*x2*x3+C',
            'C*sqrt(x1*x2/x3)+C',
            'C*x1*x2**2/sqrt(C*1-C*x3**2/x2**2+C)+C',
            'C*x1/(C*4*pi*x2*x3+C)+C',
            'C*1/(C*4*pi*x1+C)*x4*cos(C*x2+C)/x3**2+C',
            'C*3/5*x1**2/(C*4*pi*x2*x3+C)+C',
            'C*x1/x2*1/(1+x3)+C',
            'C*x1/sqrt(C*1-C*x2**2/x3**2+C)+C',
            'C*x1*x2/sqrt(C*1-C*x2**2/x3**2+C)+C',
            '-C*x1*x3*COS(C*x2+C)+C',
            '-C*x1*x2*COS(C*x3+C)+C',
            'C*sqrt(C*x1**2/x2**2-C*pi**2/x3**2+C)+C',
            'C*x1*x2*x3**2+C',
            'C*x1*x2/(C*2*pi*x3+C)+C',
            'C*x1*x2*x3/2+C',
            'C*x1*x2/(4*pi*x3)+C',
            'C*x1*(1+C*x2+C)*x3+C',
            'C*2*x1*x2/(C*x3/(2*pi)+C)+C',
            'C*sin(C*x1*x2/(C*x3/(2*pi)+C)+C)**2+C',
            'C*2*x1*(1-C*cos(C*x2*x3+C)+C)+C',
            'C*(C*x1/(2*pi)+C)**2/(C*2*x2*x3**2+C)+C',
            'C*2*pi*x3/(C*x1*x2+C)+C',
            'C*x1*(1+C*x2*cos(x3)+C)+C',
        ], 
        4: [
            # AI Faynman
            'C*exp(C*((C*x1+C*x2+C)/x3)**2+C)/(C*sqrt(C*x4+C)*x3+C)+C', 
            'C*sqrt(C*(C*x2+C*x1+C)**2+(C*x3+C*x4+C)**2+C)+C',    
            'C*x1*x2/(C*x3*x4*x2**3+C)+C',
            'C/2*x1*(C*x2**2+C*x3**2+C*x4**2+C)+C',
            'C*(C*x1-C*x2*x3+C)/sqrt(C*1-C*x2**2/x4**2+C)+C',
            'C*(C*x1-C*x3*x2/x4**2+C)/sqrt(C*1-C*x3**2/x4**2+C)+C',
            'C*(C*x1*x3+C*x2*x4+C)/(C*x1+C*x2+C)+C',
            'C*x1*x2*x3*sin(C*x4+C)+C',
            'C*1/2*x1*(C*x3**2+C*x4**2+C)*1/2*x2**2+C',
            'C*sqrt(C*x1**2+C*x2**2-C*2*x1*x2*cos(C*x3-C*x4+C))+C',
            'C*x1*x2*x3/x4+C',
            'C*4*pi*x1*(C*x2/(2*pi)+C)**2/(C*x3*x4**2+C)+C',
            'C*x1*x2*x3/x4+C',
            'C*1/(C*x1-1+C)*x2*x3/x4+C',
            'C*x1*(C*cos(C*x2*x3+C)+C*x4*cos(C*x2*x3+C)**2+C)+C',
            'C*x1/(C*4*pi*x2+C)*3*cos(C*x3+C)*sin(C*x3+C)/x4**3+C',
            'C*x1*x2/(C*x3*(C*x4**2-x5**2+C)+C)+C',
            'C*x1*x2/(C*1-C*(C*x1*x2/3+C)+C)*x3*x4+C',
            'C*1/(C*4*pi*x1*x2**2+C)*2*x3/x4+C',
            'C*x1*x2*x3/(2*x4)+C',
            'C*x1*x2*x3/x4+C',
            'C*1/(C*exp(C*(C*x1/(2*pi)+C)*x4/(C*x2*x3+C)+C)-1)+C',
            'C*(x1/(2*pi))*x2/(C*exp(C*(C*x1/(2*pi)+C)*x2/(C*x3*x4+C))-1)+C',
            'C*x1*sqrt(C*x2**2+C*x3**2+C*x4**2+C)+C',
            'C*2*x1*x2**2*x3/(C*x4/(2*pi)+C)+C',
            'C*x1*(C*exp(C*x3*x2/(C*x4*x5+C)+C)-1)+C',
            '-C*x1*x2*x3/x4+C',
        ], 
        5: [
            # AI Faynman
            'C*x1*x2*x3/(C*x4*x5*x3**3+C)+C',  
            'C*x1*(C*x2+C*x3*x4*sin(C*x5+C))+C',     
            'C*x1*x2*x3*(C*1/x4-C*1/x5+C)+C',  
            'C*x1/(2*pi)*x2**3/(pi**2*x5**2*(exp((x1/(2*pi))*x2/(x3*x4))-1))+C',   
            'C*x1*x2*x3*ln(x4/x5)+C',
            'C*x1*(C*x2-C*x3+C)*x4/x5+C',
            'C*x1*x2**2*x3/(C*3*x4*x5+C)+C',
            'C*x1/(C*4*pi*x2*x3*(1-C*x4/x5+C)+C)+C',
            'C*x1*x2*x3*x4/(C*x5/(2*pi)+C)+C',
            'C*x1/(C*exp(C*x2*x3/(C*x4*x5+C)+C)+C*exp(-C*x2*x3/(C*x4*x5+C)))+C',
            'C*x1*x2*tanh(C*x2*x3/(C*x4*x5+C)+C)+C',
            '-C*x1*x3**4/(C*2*(C*4*pi*x2+C)**2*(C*x4/(2*pi)+C)**2)*(C*1/x5**2+C)',
        ], 
        6: [
            # AI Faynman
            'C*x1*x4+C*x2*x5+C*x3*x6+C', 
            'C*x1**2*x2**2/(C*6*x3*x4*x5**3+C)+C',     
            'C*x1*exp(-C*x2*x3*x4/(C*x5*x6+C))+C',      
            'C*x1/(C*4*pi*x2+C)*3*x5/x6**5*sqrt(C*x3**2+x4**2+C)+C',
            'C*x1*(1+C*x2*x3*cos(C*x4+C)/(C*x5*x6+C)+C)+C',
            'C*(C*x1*x5*x4/(C*x6/(2*pi)+C)+C)*sin(C*(C*x2-C*x3+C)*x4/2)**2/(C*(C*x2-C*x3+C)*x4/2)**2+C',
        ], 
        7: [
            # AI Faynman
            'C*(C*1/2*x1*x4*x5**2+C)*(C*8*x6*x7**2/3+C)*(C*x2**4/(C*x2**2-C*x3**2+C)**2+C)+C',
            
        ], 
        8: [
            # AI Faynman
            'C*x1*x8/(C*x4*x5+C)+C*(C*x1*x2+C)/(C*x3*x7**2*x4*x5+C)*x6+C',            
        ], 
        9: [
            # AI Faynman
            'C*x3*x4*x5/((C*x2+C*x1+C)**2+(C*x6+C*x7+C)**2+(C*x8+C*x9)**2+C)+C',
        ], 
    }

    print(os.mkdir(folder) if not os.path.isdir(
        folder) else 'We do have the path already!')

    template = {'X': [], 'Y': 0.0, 'EQ': ''}
    fileID = 0
    # mp.set_start_method('spawn')
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
                           dataPath, fileID, time, supportPoints,
                           supportPointsTest,
                           numberofPoints,
                           trainRange, testPoints, testRange, n_levels,
                           allow_constants, const_range,
                           const_ratio, op_list, sortY, exponents,
                           threshold, templatesEQs, templateProb
                        )
                       )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
