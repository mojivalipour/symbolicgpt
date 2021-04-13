import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import column_or_1d

hidden_layer_sizes = (5, 5)
max_iter_mlp = 500000

def make_y_multi_safe(old_y, n_dims_per_input_var=1, n_dims_in_output=1):
    if isinstance(old_y, list):
        new_y = np.array(old_y)
        new_y.reshape([-1, n_dims_in_output, 1])
    else:
        new_y = old_y.copy()
    if len(new_y.shape) == 1:
        assert (n_dims_in_output == 1)
        new_y = [[[y_value] for _ in range(n_dims_per_input_var)] for y_value in new_y]
        new_y = np.array(new_y)
    elif len(new_y.shape) == 2:
        assert (n_dims_in_output == 1)
        new_y = [[y_value for _ in range(n_dims_per_input_var)] for y_value in new_y]
        new_y = np.array(new_y)
    elif new_y.shape[1] < n_dims_per_input_var:
        assert (n_dims_in_output == 1)
        new_y = [[y_value[0] for _ in range(n_dims_per_input_var)] for y_value in new_y]
        new_y = np.array(new_y)
    return new_y

class MLP_Model:
    def __init__(self):
        self.name = "MLP Model"
        self.short_name = "MLP"
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = 'adam'
        self.max_iter = max_iter_mlp
        self.warm_start = False
        self.verbose = False
        self.est_mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                    solver=self.solver,
                                    activation='relu',
                                    max_iter=self.max_iter,
                                    verbose=self.verbose,
                                    tol=1e-7,
                                    warm_start=self.warm_start,
                                    n_iter_no_change=100)

    def get_formula_string(self):
        return "(neural black box)"
        # current_inputs = ["X{}".format(i + 1) for i in range(settings.num_features)]
        # # print(current_inputs)
        # matrices = self.est_mlp.coefs_
        # vectors = self.est_mlp.intercepts_
        # for i in range(len(matrices)):
        #     current_outputs = []
        #
        #     for j in range(matrices[i].shape[1]):
        #         current_term = [vectors[i][j]]
        #         for k in range(matrices[i].shape[0]):
        #             sys.stdout.flush()
        #             current_term.append(["*", matrices[i][k][j], current_inputs[k]])
        #
        #         current_output = current_term[-1]
        #         for k in range(len(current_term), 1, -1):
        #             current_output = ["+", current_term[k - 2], current_output]
        #         current_outputs.append(current_output)
        #     current_inputs = [["max", 0, old_out] for old_out in current_outputs]
        #
        # # [-1] since we don't do relu activation on the last layer.
        # return current_inputs[0][-1]

    def get_formula(self):
        return "(neural black box)"
        # return self.get_formula_string()

    def train(self, X, Y):
        X = np.reshape(X, [X.shape[0], -1])
        Y = np.reshape(Y, [-1, 1])
        Y = column_or_1d(Y)
        self.est_mlp.fit(X, Y)
        return None

    def predict(self, X):
        return self.est_mlp.predict(X)

    # Mean square error
    def test(self, X, Y):
        X = np.array(X)
        X = np.reshape(X, [X.shape[0], -1])
        y_hat = np.reshape(self.est_mlp.predict(X), [1, -1])[0]
        y_gold = np.reshape(Y, [1, -1])[0]
        our_sum = 0
        for i in range(len(y_gold)):
            our_sum += (y_hat[i] - y_gold[i]) ** 2

        return our_sum / len(y_gold)

    def reset(self):
        self.est_mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                    solver=self.solver,
                                    activation='relu',
                                    max_iter=self.max_iter,
                                    verbose=self.verbose,
                                    tol=1e-7,
                                    warm_start=self.warm_start,
                                    n_iter_no_change=100)

    def soft_reset(self):
        self.est_mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                    solver=self.solver,
                                    activation='relu',
                                    max_iter=self.max_iter,
                                    verbose=self.verbose,
                                    tol=1e-7,
                                    warm_start=self.warm_start,
                                    n_iter_no_change=100)

    def get_simple_formula(self, digits=None):
        full_formula = self.get_formula_string()
        return full_formula
        # return DataUtils.simplify_formula(full_formula, digits=digits)

    def real_repeat_train(self, x, y=None,
                     num_repeats=1,
                     test_x=None, test_y=None,
                     verbose=True):

        # we still reduce train set size if only 1 repeat
        train_set_size = len(x)

        x = np.array(x)
        if y is not None:
            y = np.array(y)

        sample = np.random.choice(range(x.shape[0]), size=train_set_size, replace=False)
        train_x = x[sample][:]
        if y is not None:
            train_y = y[sample]
        else:
            train_y = None

        out_sample = [aaa for aaa in range(x.shape[0]) if aaa not in sample]
        valid_x = x[out_sample][:]
        if y is not None:
            valid_y = y[out_sample]
            # valid_y = self.make_y_multi_safe(valid_y)
        else:
            valid_y = None

        best_formula = ""
        best_iter = 0
        best_validation = 999999
        best_err = 999999
        old_time = time.time()

        if verbose:
            print("Beginning {} repeat sessions of {} iterations each.".format(num_repeats,
                                                                               settings.num_train_steps_in_repeat_mode))
            print()
            start_time = time.time()
            old_time = start_time

        for train_iter in range(1, 1 + num_repeats):
            if verbose:
                print("Repeated train session {} of {}.".format(train_iter, num_repeats))

            self.soft_reset()
            self.train(train_x, train_y)

            valid_err = self.test(valid_x, valid_y)

            current_time = time.time()
            if verbose:
                # print(self.get_simple_formula())
                print("Attained validation error: {:.5f}".format(valid_err))

            if valid_err < best_validation:
                best_validation = valid_err
                best_formula = self.get_simple_formula()
                best_iter = train_iter
                if test_x is not None:
                    safe_test_y = make_y_multi_safe(test_y)
                    best_err = self.test(test_x, safe_test_y)
                else:
                    best_err = valid_err
                if verbose:
                    print(">>> New best model!")
                    print(best_formula)

            if verbose:
                iters_per_minute = 60.0 / (current_time - old_time)
                print("Took {:.2f} minutes.".format((current_time - old_time) / 60))
                print("Est. {:.2f} minutes remaining.".format((num_repeats - train_iter) / iters_per_minute))
                print()
                old_time = current_time

        if verbose:
            print("Total time for repeat process: {:.2f} minutes.".format((time.time() - start_time) / 60))

        return best_formula, best_iter, best_err

    # Does not repeat train. sorry.
    def repeat_train(self, x, y=None,
                     num_repeats=1,
                     test_x=None, test_y=None,
                     verbose=True):

        # we still reduce train set size if only 1 repeat
        train_set_size = len(x)

        x = np.array(x)
        if y is not None:
            y = np.array(y)

        sample = np.random.choice(range(x.shape[0]), size=train_set_size, replace=False)
        train_x = x[sample][:]
        if y is not None:
            train_y = y[sample]
        else:
            train_y = None

        out_sample = [aaa for aaa in range(x.shape[0]) if aaa not in sample]
        valid_x = x[out_sample][:]
        if y is not None:
            valid_y = y[out_sample]
            # valid_y = self.make_y_multi_safe(valid_y)
        else:
            valid_y = None

        if verbose:
            start_time = time.time()
            old_time = start_time

        self.soft_reset()
        self.train(train_x, train_y)

        current_time = time.time()

        best_formula = self.get_simple_formula()
        if test_x is not None:
            safe_test_y = make_y_multi_safe(test_y)
            best_err = self.test(test_x, safe_test_y)
        else:
            best_err = self.test(valid_x, valid_y)
        if verbose:
            print(">>> New best model!")
            print(best_formula)

        if verbose:
            print("Took {:.2f} minutes.".format((current_time - old_time) / 60))
            print()

        if verbose:
            print("Total time for repeat process: {:.2f} minutes.".format((time.time() - start_time) / 60))

        return best_formula, 0, best_err
