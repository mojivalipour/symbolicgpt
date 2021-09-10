from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
import torch
from sympy import lambdify
import json

import omegaconf
import time

import contextlib
import numpy as np
import pandas as pd
import glob
import sys


class DummyFile(object):
    def write(self, x): pass

    def flush(self): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def relativeErr(y, y_hat, eps=1e-5):
    y_hat = np.reshape(y_hat, [1, -1])[0]
    y_gold = np.reshape(y, [1, -1])[0]

    try:
        err = ((y_hat - y_gold) ** 2) / np.linalg.norm(y_gold + eps)
    except ValueError:
        err = 100

    return np.mean(err)


def processDataFiles(files):
    text = ''""
    for f in files:
        with open(f, 'r') as h:
            lines = h.read()
            text += lines
    return text


def get_files(path, data_dir, result_dir):

    file_paths = glob.glob(path + data_dir)
    write_paths = [x.replace(".json", "_results.csv").replace(
        data_dir, result_dir) for x in file_paths]

    return file_paths, write_paths


def generate_results(file_path, model, save_path):


if __name__ == "__main__":

    raw_path = "0_1_0_13062021_174033"

    save_path = "./results/1var/{}_nesymres.csv".format(
        raw_path)

    data_path = "./data/1var/{}.json".format(raw_path)

    cfg_path = "./weights/1var/"
    architecture_path = "./weights/1var/config.yaml"

    test_data = load_metadata_hdf5(cfg_path)

    cfg = omegaconf.OmegaConf.load(architecture_path)

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(word2id=test_data.word2id,
                           id2word=test_data.id2word,
                           una_ops=test_data.una_ops,
                           bin_ops=test_data.bin_ops,
                           total_variables=list(test_data.total_variables),
                           total_coefficients=list(
                               test_data.total_coefficients),
                           rewrite_functions=list(test_data.rewrite_functions),
                           bfgs=bfgs,
                           beam_size=cfg.inference.beam_size
                           )

    weights_path = "./weights/1var/train_log_-epoch=19-val_loss=0.15.ckpt"

    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    results_data = []
    test_set = processDataFiles([file_path])
    test_set = test_set.strip().split("\n")

    for idx, sample in tqdm(enumerate(test_set)):
        t = json.loads(sample)
        X_test = torch.from_numpy(np.array(t["XT"]))
        y_test = torch.from_numpy(np.array(t["YT"]))

        X_dict = {x: X_test[:, idx].cpu()
                  for idx, x in enumerate(eq_setting["total_variables"])}

        raw_eqn = t["EQ"]
        skeleton = t["Skeleton"]

        # try:
        start_time = time.time()
        # with nostdout():
        output = fitfunc(X_test, y_test)
        equation_pred = output["best_bfgs_preds"][0]
        pred_y = lambdify(
            ",".join(test_data.total_variables),
            equation_pred)(**X_dict)
        pred_y = pred_y.numpy()
        train_time = time.time() - start_time
        err = relativeErr(y_test.numpy(), pred_y)
        print(idx, equation_pred, err)
        # except ValueError:
        #     equation_pred = "NA"
        #     pred_y = "NA",
        #     err = "NA",
        #     train_time = "NA"

        results_data.append({
            "test_index": idx,
            "true_equation": raw_eqn,
            "true_skeleton": skeleton,
            "predicted_equation": equation_pred,
            "predicted_y": pred_y,
            "rel_err": err,
            "nesymres_time": train_time
        })
    results = pd.DataFrame(results_data)
    results.to_csv(save_path, index=False)
