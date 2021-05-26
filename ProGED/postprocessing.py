# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ProGED.model_box import ModelBox

"""
Functions for the postprocessing of equation discovery results.
"""    

def models_statistics (models, data, target_variable_index = -1, success_threshold = 1e-9):
    if not isinstance(models, ModelBox):
        raise TypeError("Input to analyze_models must be a ModelBox, but was " + str(type(models)))
    
    models_keys = list(models.keys())
    
    meanpred = np.mean(data[:, target_variable_index])
    baseerror = np.sum((data[:, target_variable_index] - meanpred)**2)
    
    vmodels = {}
    logMSE = []
    RRMSE = []
    p_valid = []
    p_all = []
    logscore = []
    lowest_rrmse = []
    highest_score = []
    successNsym = 0
    successN6 = 0
    successN9 = 0
    successP9 = 0
    p_good9 = 0
    p_bad9 = 0
    N_unique_trees = 0
    
    stats_header = ["N", 
                    "N-valid",
                    "P",
                    "P-valid",
                    "N-success",
                    "P-success",
                    ]
    
    for m in models_keys:
        p_all += [models[m].p]
        
        if models[m].valid and not models[m].get_error()>=10**9:
            vmodels[m]=models[m]
            N_unique_trees += len(models[m].trees)
            
            mse = models[m].get_error()+1e-32
            logMSE += [np.log10(mse)]
            RRMSE += [np.sqrt(mse / baseerror)]
            p_valid += [models[m].p]
            logscore += [np.log10(p_valid[-1])/10 - np.log10(RRMSE[-1])]
            
            if len(lowest_rrmse) > 0:
                if RRMSE[-1] < lowest_rrmse[-1]:
                    lowest_rrmse += [RRMSE[-1]]
                else:
                    lowest_rrmse += [lowest_rrmse[-1]]
                if logscore[-1] > highest_score[-1]:
                    highest_score += [logscore[-1]]
                else:
                    highest_score += [highest_score[-1]]
            else:
                lowest_rrmse += [RRMSE[-1]]
                highest_score += [logscore[-1]]
    
            if RRMSE[-1] < success_threshold:
                p_good9 += p_valid[-1]
                successN9 += 1
                successP9 += p_valid[-1]
                # if do_symbolic:
                #     try:
                #         if symbolic_difference(solution, sp.simplify(vmodels[m].get_full_expr()), thr=9) == 0:
                #             successNsym += 1
                #     except Exception as E:
                #         print("Error in symbolic_difference, model: " + str(vmodels[m].get_full_expr()))
            else:
                p_bad9 += p_valid[-1]
            if RRMSE[-1] < 10**-6:
                successN6 += 1
                    
    MSDsortind = np.argsort(logMSE)
    score_sortind = list(reversed(np.argsort(logscore)))
    P_valid = sum(p_valid)
    P_all = sum(p_all)
    medianPgood = np.median(p_good9)
    medianPbad = np.median(p_bad9)
    medianPvalid = np.median(p_valid)
     #logmedianPgood = np.median(np.log10(p[RRMSE < 10**-9]))
     #logmedianPbad = np.median(np.log10(p[RRMSE >= 10**-9]))
     
    stats = [[len(models), len(vmodels), P_all, P_valid, successN9, successP9]]
    
    return pd.DataFrame(stats, columns = stats_header)
        