#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import tensorflow as tf

def cleanEquation(eq):
  import re
  eq = eq.replace('\n','')
  eq = re.sub(r'(?=)\[.+?\](?=)', '', eq) # remove anything between lists
  eq = eq.replace(',','')
  eq = eq.replace('<SOS_X>','')
  eq = eq.replace('<EOS_X>','')
  eq = eq.replace('<SOS_Y>','')
  eq = eq.replace('<EOS_Y>','')
  eq = eq.replace('<SOS_EQ>','')
  eq = eq.replace('<EOS_EQ>','')
  eq = eq.replace('<SOS_Skeleton>','')
  eq = eq.replace('<EOS_Skeleton>','')
  eq = eq.replace('[','')
  eq = eq.replace(']','')
  eq = eq.strip()
  return eq

# Mean square error
def mse(y, y_hat):
    y_hat = np.reshape(y_hat, [1, -1])[0]
    y_gold = np.reshape(y, [1, -1])[0]
    our_sum = 0
    for i in range(len(y_gold)):
        our_sum += (y_hat[i] - y_gold[i]) ** 2

    return our_sum / len(y_gold)

# calculate test error, show the real performance using a metric
import re
import json
import math
import demo
import numpy as np
from numpy import *
from glob import glob
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error

# config
show_found_eqns = True
min_len = 0 #@param {type:"number", min:5, max:1024, step:1}
sample_num = 1 #@param {type:"number", min:1, max:50, step:1}
top_p = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}
model_size = 'base' # @param ["large", "base", "mega"]
model_type = 'GPT2' # @param ["GPT2", "PT"]
extraName = '' #'-finetune' 
#'lm/configs/{}.json'.format(model_type) 
config_fn = 'configs/{}.json'.format(model_size) #@param {type:"string"}
#ckpt_fn = './expSymbolic_{}_{}{}_model.ckpt-524000'.format(model_type, model_size, extraName) #@param {type:"string"}
#ckpt_fn = './experimentsSymbolic_{}_model.ckpt-524000'.format(model_size) #@param {type:"string"}
ckpt_fn = 'D:/experiments/symbolicGPT2/Mesh_Simple_GPT2_256/expSymbolic_Mesh_Simple_GPT2_256_base_model.ckpt-128000'
filters = 'EQ' #@param {type:"string"} # text;
saveFlag = False #@param {type:"boolean"}

resultDict = {}
threshold = 1e5 # to handle inf or very big points

for fName in glob('D:/Datasets/Symbolic Dataset/Datasets/Mesh_Simple_GPT2/TestDataset/*.json'):
  print('Processing {}'.format(fName))
  
  if 'little' in fName:# or '0_1_0_02022021_164747.json' in fName:# or '0_5_4_02022021_164747.json' in fName: # This one was only for the development testing
    continue

  # outputName = './{}-var_{}.out'.format(re.findall(
  #                                   r'_\d_', fName.split(
  #                                   '.json')[0].split(
  #                                   '/')[-1])[0].strip('_'),
  #                                   model_type)
  outputName = '{}-var_{}.out'.format(
                            fName.split('.json')[0],
                            #re.findall(r'\d', fName)[0],
                            model_type
                          )

  with open(fName, 'r', encoding="utf-8") as f, open(outputName, 'w', encoding="utf-8") as o:
    resultDict[fName] = {'GPT2':[],
                         'MLP':[],
                         'GP':[]}

    lines = f.readlines()
    
    # <SOS_X>{}<EOS_X>
    context = ['<SOS_Y>{}<EOS_Y><SOS_EQ>'.format(
        *(np.round(val,2).tolist() for key, val in json.loads(line).items(
              ) if key == 'Y')) for line in lines]
    print(context)
    equations = demo.wraper(top_p, config_fn, ckpt_fn, min_len, sample_num, 
                            saveFlag, filters, context=context, 
                            modelType=model_type, max_num_points=30, 
                            max_num_vars=5)
    
    wrongEQCounter = 0
    yPred = []
    # compare results with other models
    from gp_model import Genetic_Model
    from mlp_model import MLP_Model

    show_found_eqns = True
    num_vars = 1
    gpm = Genetic_Model()
    mlp = MLP_Model()

    for idx, line in tqdm(enumerate(lines)):
      print("Test case {}/{}.".format(idx, len(lines)))
      
      # TODO: don't skip infinities
      if "Infinity" in line or "NaN" in line: #or i < 134:
        print('infinity or nan in input!')
        continue

      data = json.loads(line) # 50000 samples in each file

      # run the model
      #TODO: calculate the model output
      #context = ['<SOS_X>{}<EOS_X><SOS_Y>{}<EOS_Y><SOS_EQ>'.format(data['X'],data['Y'])]
      #YPred = demo.wraper(top_p, config_fn, ckpt_fn, min_len, sample_num, saveFlag, filters, context=context)

      # use Y as target labels
      Y = data['YT']

      # Evaluate YPred & Extract predicted equation
      eq = equations[idx]
      eq = cleanEquation(eq)
      yPred = []      
      YN = []
      YPredN = []
      try:
        # replace vars with values
        for xs in data['XT']:
          eqTmp = eq + '' # copy eq
          eqTmp = eqTmp.replace(' ','')
          eqTmp = eqTmp.replace('\n','')
          for i,x in enumerate(xs):
            #print('x{}'.format(i+1),x)
            # replace xi with the value in the eq
            eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
            if ',' in eqTmp:
              assert 'There is a , in the equation!'
          eqEvaluated = eval(eqTmp)
          eqEvaluated = 0 if np.isnan(eqEvaluated) else eqEvaluated
          eqEvaluated = 10000 if np.isinf(eqEvaluated) else eqEvaluated
          yPred.append(eqEvaluated)
        
        # ignore inf, or NAN
        for i, v in enumerate(Y):
          if np.isinf(Y[i]) or np.isinf(yPred[i]):
            continue
          if np.isnan(Y[i]) or np.isnan(yPred[i]):
            continue
          YN.append(Y[i])
          YPredN.append(yPred[i])
          # if v is not float('nan'): # v < threshold and 
          #   if Y[i] == np.inf:
          #     YN.append(10000)
          #   else:
          #     YN.append(Y[i])

          #   if yPred[i] == np.inf:
          #     YPredN.append(10000)
          #   else:
          #     YPredN.append(yPred[i])
      except Exception as e: #SyntaxError or AssertionError or NameError or TypeError:
        print('{} \n\n Error: {}, EQ:{}'.format(TypeError, eqTmp, eq))
        #TODO: Find a fair strategy, Resample/Ignore?!
        #continue # ignore this sample
        #yPred = np.zeros_like(Y) # no prediction
        wrongEQCounter += 1

      # ignore noisy samples with zero data on X & Y
      if len(YN) == 0:
        o.write('Test case {}/{}.\n{}\n{}: {}\n{}\n\n'.format(
          idx, len(lines),
          data['EQ'],
          model_type, "Not calculated!",
          eq
        ))
        print('Not calculated')
        continue

      dict_line = eval(line)
      print("True equation: {}".format(dict_line["EQ"]))

      # calculate rmse between YPred and Y
      #mseValue = np.log(mean_squared_error(YN,YPredN, squared=True))
      model_err = mse(YN,YPredN)
      test_err = max(np.exp(-10), model_err) 

      if show_found_eqns:
          print("{} function:  {}".format('GPT2', eq)[:550])

      print(" ---> {} Test Error: {:.5f}".format('GPT2', test_err))

      resultDict[fName]['GPT2'].append(test_err)
      o.write('Test case {}/{}.\n{}\n{}: {}\n{}'.format(
          idx, len(lines),
          data['EQ'],
          model_type, test_err,
          eq
      ))

      # tokenize to get input x, input y, and true eqn
      train_data_x = dict_line["X"]
      train_data_y = dict_line["Y"]
      test_data_x = dict_line["XT"]
      test_data_y = dict_line["YT"]
      #print("{} training points, {} test points.".format(len(train_data_x), len(test_data_x)))

      # train MLP model
      mlp.reset()
      model_eqn, _, best_err = mlp.repeat_train(train_data_x, train_data_y,
                                                test_x=test_data_x, test_y=test_data_y,                                     verbose=False)
      if show_found_eqns:
          print("{} function:  {}".format(mlp.name, model_eqn)[:550])

      # Test model on that equation
      test_err = max(np.exp(-10), best_err)  # data_utils.test_from_formula(model_eqn, test_data_x, test_data_y)
      print(" ---> {} Test Error: {:.5f}".format(mlp.short_name, test_err))

      resultDict[fName]['MLP'].append(test_err)
      o.write('\n{}: {}\n{}'.format('MLP', 
                                   test_err,
                                   model_eqn))

      # train GPL model
      gpm.reset()
      model_eqn, _, best_err = gpm.repeat_train(train_data_x, train_data_y,
                                                test_x=test_data_x, test_y=test_data_y,
                                                verbose=False)
      if show_found_eqns:
          print("{} function:  {}".format(gpm.name, model_eqn)[:550])

      # Test model on that equation
      # test_err = model.test(test_data_x, test_data_y)
      test_err = max(np.exp(-10), best_err)  # data_utils.test_from_formula(model_eqn, test_data_x, test_data_y)
      print(" ---> {} Test Error: {:.5f}".format(gpm.short_name, test_err))

      resultDict[fName]['GP'].append(test_err)
      o.write('\n{}: {}\n{}'.format('GP', 
                                   test_err,
                                   model_eqn))

      # o.write('Test case {}/{}.\n{}\n{}: {}\n{}\n\n'.format(
      #     idx, len(lines),
      #     data['Skeleton'],
      #     model_type, mseValue,
      #     eq
      # ))

      o.write('\n\n')

    print('{} of {} equations have wrong structures!'.format(wrongEQCounter, len(lines)))
    #break # for now just use one test file
  #from google.colab import files
  #files.download(outputName) 

  # save dictionary
  with open('./data.json', 'a', encoding="utf-8") as h:
    print('saving resultDict to {}'.format('./data.json'))
    json.dump(resultDict, h, ensure_ascii=False)

# plot the error frequency for model comparison
num_eqns = len(resultDict[fName]['GPT2'])
num_vars = 1

models = list(resultDict[fName].keys())
lists_of_error_scores = [resultDict[fName][key] for key in models]
linestyles = ["-","dashdot","dotted","--"]

y, x, _ = plt.hist([np.log(e) for e in lists_of_error_scores],
                   label=models,
                   cumulative=True, 
                   histtype="step", 
                   bins=2000, 
                   density="true")
plt.figure(figsize=(15, 10))

for idx, model in enumerate(models): 
  plt.plot(x[:-1], 
           y[idx] * 100, 
           linestyle=linestyles[idx], 
           label=model)

plt.legend(loc="upper left")
plt.title("{} equations of {} variables".format(num_eqns, num_vars))
plt.xlabel("Log of error")
plt.ylabel("Frequency Percentage")

plt.savefig('comparison.png')