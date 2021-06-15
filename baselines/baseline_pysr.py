import time
import json
import warnings

import numpy as np
import pandas as pd

from pysr import pysr, best, best_callable
from baseline_utils import nostdout, processDataFiles, relativeErr

warnings.filterwarnings("ignore")


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
                model = pysr(X, y, npop=100, niterations=10,
                             binary_operators=[
                                 "plus", "mult", "div", "pow", "sub"
                             ],
                             unary_operators=[
                                 "neg", "exp", "log_abs", "sqrt_abs", "sin"
                             ],
                             temp_equation_file=True,
                             verbosity=0
                             )
            equation_pred = best(model)
            model_pred = best_callable(model)
            pred_y = model_pred(X_test)
            train_time = time.time() - start_time
            err = relativeErr(y_test, pred_y)
        except ValueError:
            equation_pred = "NA"
            pred_y = "NA"
            err = "NA"
            train_time = "NA"

        results_data.append({
            "test_index": idx,
            "true_equation": raw_eqn,
            "true_skeleton": skeleton,
            "predicted_equation": equation_pred,
            "predicted_y": pred_y,
            "rel_err": err,
            "dsr_time": train_time
        })

    results = pd.DataFrame(results_data)
    results.to_csv(save_path, index=False)


if __name__ == "__main__":

    data_path = "./data/0_1_0_12062021_083325.json"
    save_path = "./results/0_1_0_12062021_083325_pysr.csv"

    generate_results(data_path, save_path)
