import contextlib
import numpy as np
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
