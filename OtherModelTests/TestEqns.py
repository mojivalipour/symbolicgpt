

# 786

from gp_model import Genetic_Model
from mlp_model import MLP_Model
import time, sys
import numpy as np

show_found_eqns = True

num_vars = 5

gpm = Genetic_Model()
mlp = MLP_Model()

gpm_error_scores = []
gpm_iter_times = []
mlp_error_scores = []
mlp_iter_times = []

# open file
input_file = open("data/{}var.txt".format(num_vars))
lines = input_file.readlines()
input_file.close()
output_file = open("output_{}var.txt".format(num_vars), "w")

print("Starting test: {} variable(s).".format(num_vars))
print()

# read lines
i = 1
num_lines = len(lines)
for line in lines:
    old_time = time.time()
    print("Test case {}/{}.".format(i, num_lines))
    output_file.write("Test case {}/{}.\n".format(i, num_lines))
    i+=1
    # TODO: don't skip infinities
    if "Infinity" in line:
        continue
    dict_line = eval(line)
    # line.replace("{", "")
    # line.replace("}", "")
    # line_parts = line.split(", ")
    print("True equation: {}".format(dict_line["EQ"]))
    output_file.write("{}\n".format(dict_line["EQ"]))


    # tokenize to get input x, input y, and true eqn
    train_data_x = dict_line["X"]
    train_data_y = dict_line["Y"]
    test_data_x = dict_line["XT"]
    test_data_y = dict_line["YT"]
    print("{} training points, {} test points.".format(len(train_data_x), len(test_data_x)))

    # train MLP model
    mlp.reset()
    itertime_start = time.time()
    model_eqn, _, best_err = mlp.repeat_train(train_data_x, train_data_y,
                                              test_x=test_data_x, test_y=test_data_y,
                                              verbose=False)
    if show_found_eqns:
        print("{} function:  {}".format(mlp.name, model_eqn)[:550])

    # Test model on that equation
    test_err = max(np.exp(-10), best_err)  # data_utils.test_from_formula(model_eqn, test_data_x, test_data_y)
    print(" ---> {} Test Error: {:.5f}".format(mlp.short_name, test_err))
    mlp_error_scores.append(test_err)
    mlp_iter_times.append(time.time() - itertime_start)
    sys.stdout.flush()
    output_file.write("{}: {}\n{}\n".format(mlp.short_name, test_err, model_eqn))
    output_file.flush()

    # train GPL model
    gpm.reset()
    itertime_start = time.time()
    model_eqn, _, best_err = gpm.repeat_train(train_data_x, train_data_y,
                                              test_x=test_data_x, test_y=test_data_y,
                                              verbose=False)
    if show_found_eqns:
        print("{} function:  {}".format(gpm.name, model_eqn)[:550])

    # Test model on that equation
    # test_err = model.test(test_data_x, test_data_y)
    test_err = max(np.exp(-10), best_err)  # data_utils.test_from_formula(model_eqn, test_data_x, test_data_y)
    print(" ---> {} Test Error: {:.5f}".format(gpm.short_name, test_err))
    gpm_error_scores.append(test_err)
    gpm_iter_times.append(time.time() - itertime_start)
    sys.stdout.flush()
    output_file.write("{}: {}\n{}\n\n".format(gpm.short_name, test_err, model_eqn))
    output_file.flush()

    iters_per_min = 60.0 / (time.time() - old_time)
    print("  Took {:.2f} minutes.".format(1.0/iters_per_min))
    print('  Est. time left:   {:.2f} minutes'.format((num_lines - i) / iters_per_min))
    print()

output_file.close()
# get