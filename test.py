#!/usr/bin/env python
# coding: utf-8
#from numpy import * # to override the math functions

def main(resultDict, modelKey):
    # set up logging
    import time
    import logging
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )

    # load libraries
    import os
    import glob
    import json
    import pickle
    import math
    import random
    import numpy as np
    from tqdm import tqdm
    
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    #from torch.utils.data import Dataset

    from utils import set_seed
    import multiprocessing
    #from multiprocessing import Process
    from matplotlib import pyplot as plt
    from trainer import Trainer, TrainerConfig
    from models import GPT, GPTConfig, PointNetConfig
    from utils import processDataFiles, CharDataset, tokenize_predict_and_evaluate, plot_and_save_results

    # set the random seed
    seed = 42
    set_seed(seed)

    # NOTE: make sure your data points are in the same range with trainRange

    # config
    numTests = 1 # number of times to generate candidates for one test equation
    parallel = False
    scratch=True # if you want to ignore the cache and start for scratch
    embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
    numPoints=[500,501] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
    numVars=3 # the dimenstion of input points x, if you don't know then use the maximum
    numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
    blockSize = 64 # spatial extent of the model for its context
    testBlockSize = 400
    batchSize = 64 # batch size of training data
    const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
    decimals = 8 # decimals of the points only if target is Skeleton
    trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton
    testRange = [[-5.0, 3.0],[-3.0, 5.0]]
    useRange = True
    dataDir = 'D:/Datasets/Symbolic Dataset/Datasets/FirstDataGenerator/'
    dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize'.format(numVars, numPoints[0], numPoints[1], embeddingSize)
    titleTemplate = "{} equations of {} variables - Benchmark"
    target = 'Skeleton' #'Skeleton' #'EQ'
    dataFolder = '3Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_500Points'
    dataTestFolder = '3Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_500Points/Test'
    addr = './SavedModels/' # where to save model
    method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
    # EMB_CAT: Concat point embedding to GPT token+pos embedding
    # EMB_SUM: Add point embedding to GPT tokens+pos embedding
    # OUT_CAT: Concat the output of the self-attention and point embedding
    # OUT_SUM: Add the output of the self-attention and point embedding
    # EMB_CON: Conditional Embedding, add the point embedding as the first token
    variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
    # NOT_VAR: Do nothing, will not pass any information from the number of variables in the equation to the GPT
    # LEA_EMB: Learnable embedding for the variables, added to the pointNET embedding
    # STR_VAR: Add the number of variables to the first token
    addVars = True if variableEmbedding == 'STR_VAR' else False
    maxNumFiles = 100 # maximum number of file to load in memory for training the neural network
    bestLoss = None # if there is any model to load as pre-trained one
    fName = '{}_SymbolicGPT_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                                'GPT_PT_{}_{}'.format(method, target), 
                                                'Padding',
                                                variableEmbedding)
    ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
    try: 
        os.mkdir(addr)
    except:
        print('Folder already exists!')

    resultDict[fName] = {modelKey:{'err':[],'trg':[],'prd':[]}}

    # load the train dataset
    train_file = 'train_dataset_{}.pb'.format(fName)
    if os.path.isfile(train_file) and not scratch:
        # just load the train set
        with open(train_file, 'rb') as f:
            train_dataset,trainText,chars = pickle.load(f)
    else:
        # process training files from scratch
        path = '{}/{}/Train/*.json'.format(dataDir, dataFolder)
        files = glob.glob(path)[:maxNumFiles]
        text = processDataFiles(files)
        chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
        text = text.split('\n') # convert the raw text to a set of examples
        trainText = text[:-1] if len(text[-1]) == 0 else text
        random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
        train_dataset = CharDataset(trainText, blockSize, chars, numVars=numVars, 
                                    numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                                    const_range=const_range, xRange=trainRange, decimals=decimals) 
        # with open(train_file, 'wb') as f:
        #     pickle.dump([train_dataset,trainText,chars], f)

    # print a random sample
    idx = np.random.randint(train_dataset.__len__())
    inputs, outputs, points, variables = train_dataset.__getitem__(idx)
    print('inputs:{}'.format(inputs))
    inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

    # load the val dataset
    path = '{}/{}/Val/*.json'.format(dataDir,dataFolder)
    files = glob.glob(path)
    textVal = processDataFiles([files[0]])
    textVal = textVal.split('\n') # convert the raw text to a set of examples
    val_dataset = CharDataset(textVal, blockSize, chars, numVars=numVars, 
                            numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                            const_range=const_range, xRange=trainRange, decimals=decimals)

    # print a random sample
    idx = np.random.randint(val_dataset.__len__())
    inputs, outputs, points, variables = val_dataset.__getitem__(idx)
    print(points.min(), points.max())
    inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

    # load the test data
    path = '{}/{}/*.json'.format(dataDir,dataTestFolder)
    files = glob.glob(path)
    textTest = processDataFiles(files)
    textTest = textTest.split('\n') # convert the raw text to a set of examples
    # test_dataset_target = CharDataset(textTest, blockSize, chars, target=target)
    test_dataset = CharDataset(textTest, testBlockSize, chars, numVars=numVars, 
                            numYs=numYs, numPoints=numPoints, addVars=addVars,
                            const_range=const_range, xRange=trainRange, decimals=decimals)

    # print a random sample
    idx = np.random.randint(test_dataset.__len__())
    inputs, outputs, points, variables = test_dataset.__getitem__(idx)
    print(points.min(), points.max())
    inputs = ''.join([train_dataset.itos[int(i)] for i in inputs])
    outputs = ''.join([train_dataset.itos[int(i)] for i in outputs])
    print('id:{}\ninputs:{}\noutputs:{}\npoints:{}\nvariables:{}'.format(idx,inputs,outputs,points, variables))

    # create the model
    pconf = PointNetConfig(embeddingSize=embeddingSize, 
                        numberofPoints=numPoints[1]-1, 
                        numberofVars=numVars, 
                        numberofYs=numYs,
                        method=method,
                        variableEmbedding=variableEmbedding)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=8, n_head=8, n_embd=embeddingSize, 
                    padding_idx=train_dataset.paddingID)
    model = GPT(mconf, pconf)
        
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=1, batch_size=batchSize, 
                        learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20, 
                        final_tokens=2*len(train_dataset)*blockSize,
                        num_workers=0, ckpt_path=ckptPath)
    trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss)

    # load the best model
    mapLocation = None if isinstance(trainer.device, int) else 'cpu'
    print(mapLocation)
    model.load_state_dict(torch.load(ckptPath, map_location=mapLocation))
    model = model.eval().to(trainer.device)
    print('{} model has been loaded!'.format(ckptPath))

    ## Test the model
    # alright, let's sample some character-level symbolic GPT 
    loader = torch.utils.data.DataLoader(
                                    test_dataset, 
                                    shuffle=False, 
                                    pin_memory=True,
                                    batch_size=1,
                                    num_workers=0)
        
    experiment_times = []
    try:
        if not parallel:
            for i, (inputs,outputs,points,variables) in tqdm(enumerate(loader), total=len(test_dataset)):
                import time
                start_time = time.time()
                tokenize_predict_and_evaluate(
                                    i, inputs, points, outputs, variables, 
                                    train_dataset, textTest, trainer, model, 
                                    resultDict, numTests, variableEmbedding, 
                                    blockSize, fName, modelKey=modelKey, device=trainer.device)
                generation_time = time.time() - start_time
                print(f'The required time for equation {i} was {generation_time}!')
                experiment_times.append(generation_time)
            plot_and_save_results(resultDict, fName, pconf, titleTemplate, 
                                textTest, modelKey=modelKey)
            print(f'The average time for one instance is {np.mean(experiment_times)}+-{np.std(experiment_times)}')
        else:
            processes = []
            device = 'cpu'
            model = model.cpu()
            #pool = multiprocessing.Pool() # processes=10
            # Define IPC manager
            manager = multiprocessing.Manager()
            # Define a list (queue) for tasks and computation results
            tasks = manager.Queue()
            results = manager.Queue()
            maximum_number_process = 10
            for i, (inputs,outputs,points,variables) in tqdm(enumerate(loader), total=len(test_dataset)):
                print(f'equation {i} ...')
                # proc = multiprocessing.Process(target=tokenize_predict_and_evaluate, args=(
                #                 i, inputs, points, outputs, variables, 
                #                 train_dataset, textTest, trainer, model, 
                #                 resultDict, numTests, variableEmbedding, 
                #                 blockSize, fName, results, modelKey,device))
                arguments = (i, inputs, points, outputs, variables, 
                             train_dataset, textTest, trainer, model, 
                             resultDict, numTests, variableEmbedding, 
                             blockSize, fName, modelKey,device)
                # Set process name
                process_name = f'Equation {i}'

                proc = multiprocessing.Process(target=parallel_wraper, 
                                               args=(process_name, tasks, results, arguments))
                
                # pool.apply_async(tokenize_predict_and_evaluate, args=(
                #                 i, inputs, points, outputs, variables, 
                #                 train_dataset, textTest, trainer, model, 
                #                 resultDict, numTests, variableEmbedding, 
                #                 blockSize, fName, modelKey,device,))
                proc.start()
                processes.append(proc)
                tasks.put(i)
                # processes.append(proc)

                if i%maximum_number_process==0:
                    print('joining ...')
                    time.sleep(1*30)

            # for proc in processes:
            #     proc.join()
            # pool.close()
            # pool.join()

            tasks.join() # wait for threads to finish
            print("All tasks completed")
            keys = ['trg','prd','err']
            items = [results.get() for _ in range(results.qsize())]
            print(items)
            for item in items:
                print(idx, item)
                for idx, key in enumerate(keys):
                    resultDict[fName][modelKey][key].append(item[idx])
            plot_and_save_results(resultDict, fName, pconf, titleTemplate, 
                                textTest, modelKey=modelKey)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    

def parallel_wraper(process_name, tasks, results, arguments):
    print('[%s] evaluation routine starts' % process_name)

    (i, inputs, points, outputs, variables, 
     train_dataset, textTest, trainer, model, 
     resultDict, numTests, variableEmbedding, 
     blockSize, fName, modelKey,device) = arguments

    while True:
        new_value = tasks.get()
        if new_value < 0:
            print(f'{process_name}:{i} evaluation routine quits')

            # Indicate finished
            results.put(-1)
            break
        else:
            from utils import tokenize_predict_and_evaluate

            eq, bestPredicted, bestErr = tokenize_predict_and_evaluate(
                i, inputs, points, outputs, variables, 
                train_dataset, textTest, trainer, model, 
                resultDict, numTests, variableEmbedding, 
                blockSize, fName, modelKey, device)

            # Output which process received the value
            # and the calculation result
            print(f'{process_name} received value: {new_value}')
            print(f'{process_name} calculated value: {(eq, bestPredicted, bestErr)}')

            # Add result to the queue
            results.put((eq, bestPredicted, bestErr))
        tasks.task_done()

    return

if __name__ == '__main__':
    modelKey='SymbolicGPT'
    resultDict = {}
    main(resultDict, modelKey)

    