import time

import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.utils.validation import column_or_1d

pop_size = 5000
generations = 20
p_crossover = 0.7
warm_start = False

def clipped_exp(x):
    return np.exp(np.clip(x, -999999, np.log(10000)))

def abs_sqrt(x):
    return np.sqrt(np.abs(x))

def abs_log(x):
    return np.log(np.sqrt(x * x + 1e-10))

exp_fn = make_function(clipped_exp, "exp", 1)
sqrt_fn = make_function(abs_sqrt, "sqrt", 1)
log_fn = make_function(abs_log, "log", 1)

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

class Genetic_Model:
    def __init__(self, n_jobs=-1):
        self.name = "Genetic Model"
        self.short_name = "GP"
        self.function_set = ["add", "mul", "div", sqrt_fn, "sin", exp_fn, log_fn]

        self.est_gp = SymbolicRegressor(population_size=pop_size,
                                        generations=generations, stopping_criteria=0.01,  # 20 gen
                                        p_crossover=p_crossover, p_subtree_mutation=0.1,
                                        p_hoist_mutation=0.05, p_point_mutation=0.1,
                                        warm_start=warm_start,
                                        max_samples=0.9, verbose=False,
                                        parsimony_coefficient=0.01,
                                        function_set=self.function_set,
                                        n_jobs=n_jobs)

    def reset(self):
        del self.est_gp
        self.est_gp = SymbolicRegressor(population_size=pop_size,
                                        generations=generations, stopping_criteria=0.01,  # 20 gen
                                        p_crossover=p_crossover, p_subtree_mutation=0.1,
                                        p_hoist_mutation=0.05, p_point_mutation=0.1,
                                        warm_start=warm_start,
                                        max_samples=0.9, verbose=False,
                                        parsimony_coefficient=0.01,
                                        function_set=self.function_set)

    def soft_reset(self):
        del self.est_gp
        self.est_gp = SymbolicRegressor(population_size=pop_size,
                                        generations=generations, stopping_criteria=0.01,  # 20 gen
                                        p_crossover=p_crossover, p_subtree_mutation=0.1,
                                        p_hoist_mutation=0.05, p_point_mutation=0.1,
                                        warm_start=warm_start,
                                        max_samples=0.9, verbose=False,
                                        parsimony_coefficient=0.01,
                                        function_set=self.function_set)

    def predict(self, X):
        return self.est_gp.predict(X)

    def get_formula(self):
        return self.est_gp._program

    def get_simple_formula(self, digits=None):
        return self.get_formula()

    def get_big_formula(self):
        formula_string = str(self.get_formula())
        nested_list_string = formula_string.replace("sqrt(", "[\'sqrt\', ")
        nested_list_string = nested_list_string.replace("add(", "[\'+\', ")
        nested_list_string = nested_list_string.replace("mul(", "[\'*\', ")
        nested_list_string = nested_list_string.replace("sub(", "[\'-\', ")
        nested_list_string = nested_list_string.replace("sin(", "[\'sin\', ")
        nested_list_string = nested_list_string.replace(")", "]")
        nested_list_string = nested_list_string.replace("X", "Y")

        retval = ""
        currently_digits = False
        current_number = ""
        for current_char in nested_list_string:
            if current_char == 'Y':
                retval += "\'x"
                currently_digits = True
                current_number = ""
            elif currently_digits:
                if current_char.isdigit():
                    # retval += "{}".format(current_char)
                    current_number += "{}".format(current_char)
                else:
                    currently_digits = False
                    retval += "{}".format(int(current_number) + 1)
                    retval += "\'{}".format(current_char)
            else:
                retval += "{}".format(current_char)

        if "Y" in retval:
            print("ERROR: formula still contains a Y...")
            print("   formula string: {}\n   nested list string: {}".format(formula_string, nested_list_string))

        return eval(retval)

    def train(self, X, Y):
        X = np.reshape(X, [X.shape[0], -1])
        Y = np.reshape(Y, [-1, 1])
        Y = column_or_1d(Y)
        self.est_gp.fit(X, Y)
        return None

    # Does not repeat train. Sorry.
    def repeat_train(self, x, y, test_x=None, test_y=None,
                     num_repeats=1,
                     num_steps_to_train=None,
                     verbose=True):
        train_set_size = len(x)
        x = np.array(x)
        y = np.reshape(np.array(y), [-1, ])
        sample = np.random.choice(range(x.shape[0]), size=train_set_size, replace=False)
        out_sample = [yyy for yyy in range(x.shape[0]) if yyy not in sample]

        train_x = x[sample][:]
        train_y = y[sample][:]
        valid_x = x[out_sample][:]
        valid_y = y[out_sample][:]

        old_time = time.time()

        if verbose:
            print("Beginning {} repeat sessions of {} iterations each.".format(num_repeats,
                                                                               settings.num_train_steps_in_repeat_mode))
            print()
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
            iters_per_minute = 60.0 / (current_time - old_time)
            print("Took {:.2f} minutes.".format((current_time - old_time) / 60))
            print("Est. {:.2f} minutes remaining.".format((num_repeats - train_iter) / iters_per_minute))
            print()

        return best_formula, 0, best_err

    # Mean square error
    def test(self, x, y):
        x = np.array(x)
        x = np.reshape(x, [x.shape[0], -1])
        y_hat = np.reshape(self.est_gp.predict(x), [1, -1])[0]
        y_gold = np.reshape(y, [1, -1])[0]
        our_sum = 0
        for i in range(len(y_gold)):
            our_sum += (y_hat[i] - y_gold[i]) ** 2

        return our_sum / len(y_gold)
