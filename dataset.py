#!/usr/bin/env python
# coding: utf-8

import os
import json
from tqdm import tqdm
import multiprocessing as mp
from GenerateData import generate_random_eqn_raw, eqn_to_str, create_dataset_from_raw_eqn, simplify_formula, dataGen

def processData(numSamples, nv, decimals, template, dataPath, fileID):
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        
        # generate a formula
        # Create a new random equation
        try:
            x,y,cleanEqn = dataGen(nv, decimals)
        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i-1
            continue
        
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

def main():
    # Config
    numVars = list(range(31)) #[1,2,3,4,5]
    decimals = 2
    numSamples = 1000000 # number of generated samples
    folder = './Dataset'
    dataPath = folder +'/{}_{}.json'

    print(os.mkdir(folder) if not os.path.isdir(folder) else 'We do have the path already!')

    template = {'X':[], 'Y':0.0, 'EQ':''}
    fileID = 0
    #mp.set_start_method('spawn')
    #q = mp.Queue()
    processes = []
    for nv in numVars:
        print('Processing equations with {} variables!'.format(nv))
        p = mp.Process(target=processData, args=(numSamples, nv, decimals, template, dataPath, fileID,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()


