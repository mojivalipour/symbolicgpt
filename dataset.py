#!/usr/bin/env python
# coding: utf-8

import os
import json
from tqdm import tqdm
from GenerateData import generate_random_eqn_raw, eqn_to_str, create_dataset_from_raw_eqn, simplify_formula

# Config
numVars = [1,2,3,4,5]
decimals = 2
numSamples = 100000 # number of generated samples
folder = './Dataset'
dataPath = folder +'/{}_{}.json'

print(os.mkdir(folder) if not os.path.isdir(folder) else 'We do have the path already!')

template = {'X':[], 'Y':0.0, 'EQ':''}

fileID = 0
for nv in numVars:
    print('Processing equations with {} variables!'.format(nv))
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        
        # generate a formula
        # Create a new random equation
        currEqn = generate_random_eqn_raw(n_vars=nv)
        cleanEqn = eqn_to_str(currEqn)
        data = create_dataset_from_raw_eqn(currEqn, n_points=1, n_vars=nv, decimals=decimals)
        x,y = data[0]
        
        # hold data in the structure
        structure['X'] = x
        structure['Y'] = y
        structure['EQ'] = cleanEqn
        
        outputPath = dataPath.format(fileID, nv)
        if os.path.exists(outputPath):
            fileSize = os.path.getsize(outputPath)
            if fileSize > 500000000: # 500 MB
                fileID +=1 
        with open(outputPath, "a", encoding="utf-8") as h:
            json.dump(structure, h, ensure_ascii=False)
            h.write('\n')



