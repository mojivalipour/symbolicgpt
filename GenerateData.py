''' References:
- https://stackoverflow.com/questions/492519/timeout-on-a-function-call
'''

import sys
import sympy
#import threading
import numpy as np
from sympy import sympify, expand
from wrapt_timeout_decorator import *

# try:
#     import thread
# except ImportError:
#     import _thread as thread

main_op_list = ["id", "add", "mul", "div", "sqrt", "sin", "exp", "log"]

eps = 1e-4
big_eps = 1e-3

# def quit_function(fn_name):
#     # print to stderr, unbuffered in Python 2.
#     #TODO: find a cleaner way, using sys without importing is going to raise an error (it seems this is the only way for now)
#     print('{0} took too long'.format(fn_name), file=sys.stderr)
#     raise KeyboardInterrupt
#     sys.stderr.flush() # Python 3 stderr is likely buffered.
#     thread.interrupt_main() # raises KeyboardInterrupt

# def exit_after(s):
#     '''
#     use as decorator to exit process if
#     function takes longer than s seconds
#     '''
#     def outer(fn):
#         def inner(*args, **kwargs):
#             timer = threading.Timer(s, quit_function, args=[fn.__name__])
#             timer.start()
#             try:
#                 result = fn(*args, **kwargs)
#             finally:
#                 timer.cancel()
#             return result
#         return inner
#     return outer

def safe_abs(x):
    return np.sqrt(x * x + eps)

def safe_div(x, y):
    return np.sign(y) * x / safe_abs(y)

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

"""
# Function to generate random equation as operator/input list
# Variables are numbered 1 ... n, and 0 does not appear
# Constants appear as [float] e.g [3.14]
def generate_random_eqn_raw(n_levels=2, n_vars=2, op_list=main_op_list,
                            allow_constants=True, const_min_val=-4.0, const_max_val=4.0):
    eqn_ops = list(np.random.choice(op_list, size=int(2**n_levels)-1, replace=True))

    if allow_constants:
        eqn_vars = list(np.random.choice(range(1, max(int(n_vars * 1.6), n_vars+2)),
                                         size=int(2**n_levels), replace=True))
        for i in range(len(eqn_vars)):
            if eqn_vars[i] >= n_vars + 1:
                eqn_vars[i] = [np.random.uniform(const_min_val, const_max_val)]
    else:
        eqn_vars = list(np.random.choice(range(1, 1+n_vars), size=int(2 ** n_levels), replace=True))
    return [eqn_ops, eqn_vars]
"""


# Function to generate random equation as operator/input list and weight/bias list
# Variables are numbered 1 ... n, and 0 does not appear
# Constants appear in weight and bias lists.
# const_ratio determines how many weights are not 1, and how many biases are not 0
def generate_random_eqn_raw(n_levels=2, n_vars=2, op_list=main_op_list,
                            allow_constants=True, const_min_val=-4.0, const_max_val=4.0,
                            const_ratio=0.8):
    eqn_ops = list(np.random.choice(op_list, size=int(2**n_levels)-1, replace=True))
    eqn_vars = list(np.random.choice(range(1, (n_vars + 1)), size=int(2 ** n_levels), replace=True))
    max_bound = max(np.abs(const_min_val), np.abs(const_max_val))
    eqn_weights = list(np.random.uniform(-1 * max_bound, max_bound, size=len(eqn_vars)))
    eqn_biases = list(np.random.uniform(-1 * max_bound, max_bound, size=len(eqn_vars)))

    if not allow_constants:
        const_ratio = 0.0
    random_const_chooser_w = np.random.uniform(0, 1, len(eqn_weights))
    random_const_chooser_b = np.random.uniform(0, 1, len(eqn_biases))

    for i in range(len(eqn_weights)):
        if random_const_chooser_w[i] >= const_ratio:
            eqn_weights[i] = 1
        if random_const_chooser_b[i] >= const_ratio:
            eqn_biases[i] = 0

    return [eqn_ops, eqn_vars, eqn_weights, eqn_biases]


# Function to create a data set given an operator/input list
# Output is a list with entries of the form of pairs [ [x1, ..., xn], y ]
def create_dataset_from_raw_eqn(raw_eqn, n_points, n_vars=2,
                                min_x=0.1, max_x=3.1,
                                noise_std_dev=0, decimals=2, 
                                supportPoints=None):
    x_data = [list(np.round(np.random.uniform(min_x, max_x, n_vars), decimals)) for _ in range(n_points)] if supportPoints is None else list(supportPoints)
    #y_data = [np.round(np.log(evaluate_eqn_list_on_datum(raw_eqn, x_data_i) + np.random.normal(0, noise_std_dev)), decimals)
    #          for x_data_i in x_data]
    y_data = []
    for x_data_i in x_data:
        y = evaluate_eqn_list_on_datum(raw_eqn, x_data_i) + np.random.normal(0, noise_std_dev)    
        #sign = np.sign(y)
        #y = sign * np.log(np.abs(y)) #
        y = np.round(y, decimals)
        y_data.append(y)
    #[[list(x_data[i]), y_data[i]] for i in range(len(y_data))]

    return x_data, y_data

# Function to evaluate equation (in list format) on a data point
def evaluate_eqn_list_on_datum(raw_eqn, input_x):
    eqn_ops = raw_eqn[0]
    eqn_vars = raw_eqn[1]
    eqn_weights = raw_eqn[2]
    eqn_biases = raw_eqn[3]
    current_op = eqn_ops[0]

    if len(eqn_ops) == 1:
        left_side = eqn_weights[0] * input_x[eqn_vars[0] - 1] + eqn_biases[0]
        right_side = eqn_weights[1] * input_x[eqn_vars[1] - 1] + eqn_biases[1]

    else:
        split_point = int((len(eqn_ops) + 1) / 2)
        left_ops = eqn_ops[1:split_point]
        right_ops = eqn_ops[split_point:]

        left_vars = eqn_vars[:split_point]
        right_vars = eqn_vars[split_point:]

        left_weights = eqn_weights[:split_point]
        right_weights = eqn_weights[split_point:]

        left_biases = eqn_biases[:split_point]
        right_biases = eqn_biases[split_point:]

        left_side = evaluate_eqn_list_on_datum([left_ops, left_vars, left_weights, left_biases], input_x)
        right_side = evaluate_eqn_list_on_datum([right_ops, right_vars, right_weights, right_biases], input_x)

    if current_op == 'id':
        return left_side

    if current_op == 'sqrt':
        return np.sqrt(np.abs(left_side))

    if current_op == 'log':
        return np.log(np.sqrt(left_side * left_side + 1e-10))

    if current_op == 'sin':
        return np.sin(left_side)

    if current_op == 'exp':
        return np.exp(left_side)

    if current_op == 'add':
        return left_side + right_side

    if current_op == 'mul':
        return left_side * right_side

    if current_op == 'sub':
        return left_side - right_side

    if current_op == 'div':
        return safe_div(left_side, right_side)

    return None

def raw_eqn_to_str(raw_eqn, n_vars=2):
    eqn_ops = raw_eqn[0]
    eqn_vars = raw_eqn[1]
    eqn_weights = raw_eqn[2]
    eqn_biases = raw_eqn[3]
    current_op = eqn_ops[0]

    if len(eqn_ops) == 1:
        if n_vars > 1:
            left_side = "({} * x{} + {})".format(eqn_weights[0], eqn_vars[0], eqn_biases[0])
            right_side = "({} * x{} + {})".format(eqn_weights[1], eqn_vars[1], eqn_biases[1])
        else:
            left_side = "({} * x + {})".format(eqn_weights[0], eqn_biases[0])
            right_side = "({} * x + {})".format(eqn_weights[1], eqn_biases[1])

    else:
        split_point = int((len(eqn_ops) + 1) / 2)
        left_ops = eqn_ops[1:split_point]
        right_ops = eqn_ops[split_point:]

        left_vars = eqn_vars[:split_point]
        right_vars = eqn_vars[split_point:]

        left_side = eqn_to_str([left_ops, left_vars])
        right_side = eqn_to_str([right_ops, right_vars])

    left_is_float = False
    right_is_float = False
    left_value = np.nan
    right_value = np.nan

    if is_float(left_side):
        left_value = float(left_side)
        left_is_float = True
    if is_float(right_side):
        right_value = float(right_side)
        right_is_float = True

    if current_op == 'id':
        return left_side

    if current_op == 'sqrt':
        if left_is_float:
            return "{:.3f}".format(np.sqrt(np.abs(left_value)))
        return "sqrt({})".format(left_side)

    if current_op == 'log':
        if left_is_float:
            return "{:.3f}".format(np.math.log(safe_abs(left_value)))
        return "log({})".format(left_side)

    if current_op == 'sin':
        if left_is_float:
            return "{:.3f}".format(np.sin(left_value))
        return "sin({})".format(left_side)

    if current_op == 'exp':
        if left_is_float:
            return "{:.3f}".format(np.exp(left_value))
        return "exp({})".format(left_side)

    if current_op == 'add':
        if left_is_float and right_is_float:
            return "{:.3f}".format(left_value + right_value)
        return "({} + {})".format(left_side, right_side)

    if current_op == 'mul':
        if left_is_float and right_is_float:
            return "{:.3f}".format(left_value * right_value)
        return "({} * {})".format(left_side, right_side)

    if current_op == 'sub':
        if left_is_float and right_is_float:
            return "{:.3f}".format(left_value - right_value)
        return "({} - {})".format(left_side, right_side)

    if current_op == 'div':
        if left_is_float and right_is_float:
            return "{:.3f}".format(safe_div(left_value, right_value))
        return "({} / {})".format(left_side, right_side)

    return None

# @func_set_timeout(5)
# def timing(x):
#     return sympy.preorder_traversal(x)

def simplify_formula(formula_to_simplify, digits=4):
    if len("{}".format(formula_to_simplify)) > 1000:
        return "{}".format(formula_to_simplify)
    orig_form_str = sympify(formula_to_simplify)
    if len("{}".format(orig_form_str)) > 1000:
        return "{}".format(orig_form_str)

    if len("{}".format(orig_form_str)) < 700:
        orig_form_str = expand(orig_form_str)

    rounded = orig_form_str

    traversed = sympy.preorder_traversal(orig_form_str)
    # try:
    #     traversed = timing(orig_form_str) #ft(5, timing, kargs={'x':orig_form_str})
    # except FunctionTimedOut:
    #     print("sympy.preorder_traversal(orig_form_str) could not complete within 5 seconds and was terminated.\n")
    # except Exception as e:
    #     # Handle any exceptions that timing might raise here
    #     print("sympy.preorder_traversal(orig_form_str) was terminated.\n")
    #     return False

    for a in traversed:
        if isinstance(a, sympy.Float):
            if digits is not None:
                if np.abs(a) < 10**(-1*digits):
                    rounded = rounded.subs(a, 0)
                else:
                    rounded = rounded.subs(a, round(a, digits))
            elif np.abs(a) < big_eps:
                rounded = rounded.subs(a, 0)

    return "{}".format(rounded)

def eqn_to_str(raw_eqn, n_vars=2, decimals=2):
    return simplify_formula(raw_eqn_to_str(raw_eqn, n_vars), digits=decimals)

@timeout(5) #, use_signals=False)
def dataGen(nv, decimals, supportPoints=None):
    nPoints = 1 if supportPoints is None else len(supportPoints)
    currEqn = generate_random_eqn_raw(n_vars=nv)
    cleanEqn = eqn_to_str(currEqn, n_vars=nv, decimals=decimals)
    data = create_dataset_from_raw_eqn(currEqn, n_points=nPoints, n_vars=nv, decimals=decimals, supportPoints=supportPoints)
    return data[0], data[1], cleanEqn

######################################
# Use cases
######################################

# # Create a new random equation
# curr_eqn = generate_random_eqn_raw()
# clean_eqn = eqn_to_str(curr_eqn)
# print(clean_eqn)
#
# # Create data for that equation
# data = create_dataset_from_raw_eqn(curr_eqn, n_points=5)
# print(data)

# print(dataGen(4, 3))
