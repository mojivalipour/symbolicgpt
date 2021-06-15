import time
import json
import warnings

import numpy as np
import pandas as pd

from dsr import DeepSymbolicRegressor
from baseline_utils import nostdout, processDataFiles, relativeErr

warnings.filterwarnings("ignore")


def generate_results(file_path, config_path, save_path):

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
                model = DeepSymbolicRegressor(config_path)
                model.fit(X, y)
                equation_pred = model.program_.pretty()
                pred_y = model.predict(X_test)
            train_time = time.time() - start_time
            err = relativeErr(y_test, pred_y)
            predicted_tree = model.program_
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
    config_path = "./dsr_baseline_config.json"
    save_path = "./results/0_1_0_12062021_083325_dsr.csv"

    generate_results(data_path, config_path, save_path)
