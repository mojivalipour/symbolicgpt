import time
import json
import warnings

import numpy as np
import pandas as pd

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from baseline_utils import nostdout, processDataFiles, relativeErr

warnings.filterwarnings("ignore")

POP_SIZE = 1000
GENERATIONS = 10
P_CROSSOVER = 0.7
WARM_START = False


def clipped_exp(x):

    return np.exp(np.clip(x, -999999, np.log(10000)))


def abs_sqrt(x):
    x = x.astype(float)
    return np.sqrt(np.abs(x))


def abs_log(x):
    x = x.astype(float)
    return np.log(np.sqrt(x * x + 1e-10))


exp_fn = make_function(clipped_exp, "exp", 1)
sqrt_fn = make_function(abs_sqrt, "sqrt", 1)
log_fn = make_function(abs_log, "log", 1)

FUNCTION_SET = ["add", "mul", "div", sqrt_fn, "sin", exp_fn, log_fn]


def generate_results(file_path, save_path):

    results_data = []
    test_set = processDataFiles([file_path])
    test_set = test_set.strip().split("\n")

    for idx, sample in enumerate(test_set):
        t = json.loads(sample)
        X = np.array(t["X"])
        y = np.array(t["Y"])
        X_test = np.array(t["XT"])
        y_test = np.array(t["YT"])
        raw_eqn = t["EQ"]
        skeleton = t["Skeleton"]

        try:
            start_time = time.time()
            with nostdout():
                model = SymbolicRegressor(
                    population_size=POP_SIZE,
                    generations=GENERATIONS, stopping_criteria=0.01,  # 20 gen
                    p_crossover=P_CROSSOVER, p_subtree_mutation=0.1,
                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                    warm_start=WARM_START,
                    max_samples=0.9, verbose=False,
                    parsimony_coefficient=0.01,
                    function_set=FUNCTION_SET,
                    n_jobs=-1
                )
                model.fit(X, y)
                equation_pred = model._program
                pred_y = model.predict(X_test)
            train_time = time.time() - start_time
            err = relativeErr(y_test, pred_y)
            predicted_tree = model._program
        except ValueError:
            equation_pred = "NA"
            predicted_tree = "NA"
            pred_y = "NA"
            err = "NA"
            train_time = "NA"

        results_data.append({
            "test_index": idx,
            "true_equation": raw_eqn,
            "true_skeleton": skeleton,
            "predicted_equation": equation_pred,
            "predicted_tree": predicted_tree,
            "predicted_y": pred_y,
            "rel_err": err,
            "dsr_time": train_time
        })

    results = pd.DataFrame(results_data)
    results.to_csv(save_path, index=False)


if __name__ == "__main__":

    data_path = "./data/0_1_0_12062021_083325.json"
    save_path = "./results/0_1_0_12062021_083325_gp.csv"

    generate_results(data_path, save_path)
