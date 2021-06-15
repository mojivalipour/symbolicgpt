''' References:
- https://stackoverflow.com/questions/492519/timeout-on-a-function-call
'''

import re
import random
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

seed = 2021  # 2021 Train, 2022 Val, 2023 Test
random.seed(seed)
np.random.seed(seed=seed)  # we didn't use this line for the training data

# main_op_list = ["id", "add", "mul", "div", "sqrt", "sin", "exp", "log"]

eps = 1e-4
big_eps = 1e-3

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

def generate_random_eqn_raw(n_levels=2, n_vars=2, op_list=['id', 'sin'],
                            allow_constants=True, const_range=[-0.4, 0.4],
                            const_ratio=0.8, exponents=[3, 4, 5, 6]):

    level_to_use = np.random.randint(1, n_levels)
    const_min_val, const_max_val = const_range
    eqn_ops = list(np.random.choice(
        op_list, size=int(2**level_to_use) - 1, replace=True))
    eqn_vars = list(np.random.choice(range(1, (n_vars + 1)),
                    size=int(2 ** level_to_use), replace=True))
    max_bound = max(np.abs(const_min_val), np.abs(const_max_val))
    eqn_weights = list(np.random.uniform(-1 * max_bound,
                       max_bound, size=len(eqn_vars)))
    eqn_biases = list(np.random.uniform(-1 * max_bound,
                      max_bound, size=len(eqn_vars)))
    exponent_list = [exponents[np.random.randint(
        len(exponents))] for i in range(2 ** level_to_use)]

    if not allow_constants:
        const_ratio = 0.0
    random_const_chooser_w = np.random.uniform(0, 1, len(eqn_weights))
    random_const_chooser_b = np.random.uniform(0, 1, len(eqn_biases))

    for i in range(len(eqn_weights)):
        if random_const_chooser_w[i] >= const_ratio:
            eqn_weights[i] = 1
        if random_const_chooser_b[i] >= const_ratio:
            eqn_biases[i] = 0

    return [eqn_ops, eqn_vars, eqn_weights, eqn_biases, exponent_list]

# Function to create a data set given an operator/input list
# Output is a list with entries of the form of pairs [ [x1, ..., xn], y ]
def create_dataset_from_raw_eqn(raw_eqn, n_points, n_vars=2,
                                min_x=0.1, max_x=3.1,
                                noise_std_dev=0, decimals=2,
                                supportPoints=None):
    if isinstance(min_x, list):
        # support multiple range
        x_data = []
        for minix, maxix in zip(min_x, max_x):
            x_data += [list(np.round(np.random.uniform(minix, maxix, n_vars), decimals))
                for _ in range(n_points)] if supportPoints is None else list(supportPoints)
    else:
        x_data = [list(np.round(np.random.uniform(min_x, max_x, n_vars), decimals))
                for _ in range(n_points)] if supportPoints is None else list(supportPoints)
    
    # y_data = [np.round(np.log(evaluate_eqn_list_on_datum(raw_eqn, x_data_i) + np.random.normal(0, noise_std_dev)), decimals)
    #          for x_data_i in x_data]
    y_data = []
    for x_data_i in x_data:
        y = evaluate_eqn_list_on_datum(
            raw_eqn, x_data_i) + np.random.normal(0, noise_std_dev)
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
    exponent_list = raw_eqn[4]
    current_op = eqn_ops[0]

    exponent = exponent_list[0]
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

        left_exponent = exponent_list[:split_point]
        right_exponent = exponent_list[split_point:]

        left_side = evaluate_eqn_list_on_datum(
            [left_ops, left_vars, left_weights, left_biases, left_exponent], input_x)
        right_side = evaluate_eqn_list_on_datum(
            [right_ops, right_vars, right_weights, right_biases, right_exponent], input_x)

    if current_op == 'id':
        return left_side

    if current_op == 'sqrt':
        return np.sqrt(np.abs(left_side))

    if current_op == 'pow':
        return np.power(left_side, exponent)

    if current_op == 'log':
        return np.log(np.sqrt(left_side * left_side + 1e-10))

    if current_op == 'sin':
        return np.sin(left_side)

    if current_op == 'cos':
        return np.cos(left_side)

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
    exponent_list = raw_eqn[4]
    current_op = eqn_ops[0]

    exponent = exponent_list[0]
    if len(eqn_ops) == 1:
        # if n_vars > 1:
        left_side = "({}*x{}+{})".format(
            float(eqn_weights[0]), eqn_vars[0], float(eqn_biases[0]))
        right_side = "({}*x{}+{})".format(
            float(eqn_weights[1]), eqn_vars[1], float(eqn_biases[1]))

        # else:
        #     left_side = "({}*x+{})".format(
        #         float(eqn_weights[0]), float(eqn_biases[0]))
        #     right_side = "({}*x+{})".format(
        #         float(eqn_weights[1]), float(eqn_biases[1]))

    else:
        split_point = int((len(eqn_ops) + 1) / 2)
        left_ops = eqn_ops[1:split_point]
        right_ops = eqn_ops[split_point:]

        left_vars = eqn_vars[:split_point]
        right_vars = eqn_vars[split_point:]

        left_weights = eqn_weights[:split_point]
        left_biases = eqn_biases[:split_point]

        right_weights = eqn_weights[split_point:]
        right_biases = eqn_biases[split_point:]

        left_exponent = exponent_list[:split_point]
        right_exponent = exponent_list[split_point:]

        left_side = eqn_to_str(
            [left_ops, left_vars, left_weights, left_biases, left_exponent])
        right_side = eqn_to_str(
            [right_ops, right_vars, right_weights, right_biases, right_exponent])

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
        return "sqrt(abs({}))".format(left_side)

    if current_op == 'log':
        if left_is_float:
            return "{:.3f}".format(np.math.log(safe_abs(left_value)))
        return "log({})".format(left_side)

    if current_op == 'sin':
        if left_is_float:
            return "{:.3f}".format(np.sin(left_value))
        return "sin({})".format(left_side)

    if current_op == 'pow':
        # exponent = exponents[np.random.randint(len(exponents))]
        if left_is_float:
            return "{:.3f}".format(np.power(left_value, exponent))
        # return "pow({},{})".format(left_side, exponent)
        return "({}**{})".format(left_side, exponent)

    if current_op == 'cos':
        if left_is_float:
            return "{:.3f}".format(np.cos(left_value))
        return "cos({})".format(left_side)

    if current_op == 'exp':
        if left_is_float:
            return "{:.3f}".format(np.exp(left_value))
        return "exp({})".format(left_side)

    if current_op == 'add':
        if left_is_float and right_is_float:
            return "{:.3f}".format(left_value + right_value)
        return "({}+{})".format(left_side, right_side)

    if current_op == 'mul':
        if left_is_float and right_is_float:
            return "{:.3f}".format(left_value * right_value)
        return "({}*{})".format(left_side, right_side)

    if current_op == 'sub':
        if left_is_float and right_is_float:
            return "{:.3f}".format(left_value - right_value)
        return "({}-{})".format(left_side, right_side)

    if current_op == 'div':
        if left_is_float and right_is_float:
            return "{:.3f}".format(safe_div(left_value, right_value))
        return "({}/{})".format(left_side, right_side)

    return None

# return not only skeleton but also the placeholder of constant as fixed

def raw_eqn_to_skeleton_structure(raw_eqn, n_vars=2):
    eqn_ops = raw_eqn[0]
    eqn_vars = raw_eqn[1]
    eqn_weights = raw_eqn[2]
    eqn_biases = raw_eqn[3]
    exponent_list = raw_eqn[4]
    current_op = eqn_ops[0]

    exponent = exponent_list[0]
    if len(eqn_ops) == 1:
        # if n_vars > 1:
        left_side = "(C*x{}+C)".format(eqn_vars[0])
        right_side = "(C*x{}+C)".format(eqn_vars[1])

        # else:
        #     left_side = "(C*x+C)".format(eqn_weights[0], eqn_biases[0])
        #     right_side = "(C*x+C)".format(eqn_weights[1], eqn_biases[1])

    else:
        split_point = int((len(eqn_ops) + 1) / 2)
        left_ops = eqn_ops[1:split_point]
        right_ops = eqn_ops[split_point:]

        left_vars = eqn_vars[:split_point]
        right_vars = eqn_vars[split_point:]

        left_weights = eqn_weights[:split_point]
        left_biases = eqn_biases[:split_point]

        right_weights = eqn_weights[split_point:]
        right_biases = eqn_biases[split_point:]

        left_exponent = exponent_list[:split_point]
        right_exponent = exponent_list[split_point:]

        left_side = eqn_to_str_skeleton_structure(
            [left_ops, left_vars, left_weights, left_biases, left_exponent])
        right_side = eqn_to_str_skeleton_structure(
            [right_ops, right_vars, right_weights, right_biases, right_exponent])

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
        if left_is_float:
            return "C"
        return left_side

    if current_op == 'sqrt':
        if left_is_float:
            return "C*sqrt(C)"
        return "sqrt({})".format(left_side)

    if current_op == 'log':
        if left_is_float:
            return "C*log(C)"
        return "log({})".format(left_side)

    if current_op == 'sin':
        if left_is_float:
            return "C*sin(C)"
        return "sin({})".format(left_side)

    if current_op == 'pow':
        # exponent = exponents[np.random.randint(len(exponents))]
        if left_is_float:
            return "C*(C**{})".format(exponent)
        # return "pow({},{})".format(left_side, exponent)
        return "({}**{})".format(left_side, exponent)

    if current_op == 'cos':
        if left_is_float:
            return "C*cos(C)"
        return "cos({})".format(left_side)

    if current_op == 'exp':
        if left_is_float:
            return "C*exp(C)"
        return "exp({})".format(left_side)

    if current_op == 'add':
        if left_is_float and right_is_float:
            return "(C+C)"
        elif left_is_float:
            return "(C+{})".format(right_side)
        elif right_is_float:
            return "({}+C)".format(left_side)
        return "({}+{})".format(left_side, right_side)

    if current_op == 'mul':
        if left_is_float and right_is_float:
            return "(C*C)"
        elif left_is_float:
            return "(C*{})".format(right_side)
        elif right_is_float:
            return "({}*C)".format(left_side)
        return "({}*{})".format(left_side, right_side)

    if current_op == 'sub':
        if left_is_float and right_is_float:
            return "(C-C)"
        elif left_is_float:
            return "(C-{})".format(right_side)
        elif right_is_float:
            return "({}-C)".format(left_side)
        return "({}-{})".format(left_side, right_side)

    if current_op == 'div':
        if left_is_float and right_is_float:
            return "(C/C)"
        elif left_is_float:
            return "(C/{})".format(right_side)
        elif right_is_float:
            return "({}/C)".format(left_side)
        return "({}/{})".format(left_side, right_side)

    return None

# @func_set_timeout(5)
# def timing(x):
#     return sympy.preorder_traversal(x)

def simplify_formula(formula_to_simplify, digits=4):
    # if len("{}".format(formula_to_simplify)) > 1000:
    #     return "{}".format(formula_to_simplify)
    orig_form_str = sympify(formula_to_simplify)
    # if len("{}".format(orig_form_str)) > 1000:
    #     return "{}".format(orig_form_str)

    # if len("{}".format(orig_form_str)) < 700:
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

    try:
        for a in traversed:
            if isinstance(a, sympy.Float):
                if digits is not None:
                    if np.abs(a) < 10**(-1 * digits):
                        rounded = rounded.subs(a, 0)
                    else:
                        rounded = rounded.subs(a, round(a, digits))
                elif np.abs(a) < big_eps:
                    rounded = rounded.subs(a, 0)
    except:
        return None

    return "{}".format(rounded).replace(' ', '')

def eqn_to_str(raw_eqn, n_vars=2, decimals=2):
    return simplify_formula(raw_eqn_to_str(raw_eqn, n_vars), digits=decimals)

def eqn_to_str_skeleton_structure(raw_eqn, n_vars=2, decimals=2):
    return raw_eqn_to_skeleton_structure(raw_eqn, n_vars)

# receive an string equation and convert any number to a specific constant token
def eqn_to_str_skeleton(eq):
    #constants = re.findall("\d+\.\d+", eq)  # replace float number first
    #for c in constants:
    #    eq = eq.replace(c, 'C')
    eq = re.sub("\d+\.\d+", "C", eq)
    eq = eq.replace('-', '+')
    eq = re.sub(r'^\+', '', eq)

    # TODO: find a way to resolve standalone numbers, not x1 etc. The easiest way is to make sure every constant is float.

    eq = eq + '+C'  # add a bias
    dic = {
        '(+': '(',
        'sin(': 'sin(C*',
        'cos(': 'cos(C*',
        'log(': 'log(C*',
        'exp(': 'exp(C*',
        'sqrt(': 'sqrt(C*',
        'sin': 'C*sin',
        'cos': 'C*cos',
        'log': 'C*log',
        'exp': 'C*exp',
        'abs': 'C*abs',
        'sqrt': 'C*sqrt',
        # '+':'*C+C*',
        # '-':'*C-C*',
        # '*':'*C*',
        # '/':'*C/',
        'C**2': 'C',
        'C**3': 'C',
        'C**4': 'C',
        'C**5': 'C',
        'C**6': 'C',
        'C**7': 'C',
        'C**8': 'C',
        'C**9': 'C',
        'C*C*C': 'C',
        'C*C': 'C',  # remove the duplicates
        'C+C': 'C'  # handles the case when it is -1.23x1 converts to C * x1
    }
    for k in dic:
        eq = eq.replace(k, dic[k])
    return eq

#@timeout(5) #, use_signals=False)
def dataGen(nv, decimals, 
            numberofPoints=[0, 10],
            supportPoints=None,
            supportPointsTest=None,
            xRange=[0.1, 3.1], testPoints=False,
            testRange=[0.0, 6.0],
            n_levels=3,
            allow_constants=True,
            const_range=[-0.4, 0.4],
            const_ratio=0.8,
            op_list=[
                "id", "add", "mul", "div",
                "sqrt", "sin", "exp", "log"],
            exponents=[2, 3]):
    """
    - nv: Number of Variables
    - decimals: number of floating points
    - numberofPoints: number of points for each instance
    - seed: random generator seed (DEPRICATED)
    - xRange: range of x
    - testPoints: a flag to generate (XT) for testing range
    - testRange: range of x for testing
    - n_levels: complexity of formulas
    - op_list: operator lists for using in the experiment
    """
    # nPoints = np.random.randint(
    #     *numberofPoints) if supportPoints is None else len(supportPoints)
    currEqn = generate_random_eqn_raw(n_vars=nv,
                                      n_levels=n_levels,
                                      op_list=op_list,
                                      allow_constants=allow_constants,
                                      const_range=const_range,
                                      const_ratio=const_ratio,
                                      exponents=exponents)

    cleanEqn = eqn_to_str(currEqn, n_vars=nv,
                          decimals=decimals)
    # skeletonEqn = eqn_to_str_skeleton_structure(
    # currEqn, n_vars=nv, decimals=decimals)
    skeletonEqn = eqn_to_str_skeleton(cleanEqn)
    # data = create_dataset_from_raw_eqn(currEqn, n_points=nPoints, n_vars=nv,
    #                                    decimals=decimals, supportPoints=supportPoints, min_x=xRange[0], max_x=xRange[1])
    # if testPoints:
    #     dataTest = create_dataset_from_raw_eqn(currEqn, n_points=nPoints, n_vars=nv, decimals=decimals,
    #                                            supportPoints=supportPointsTest, min_x=testRange[0], max_x=testRange[1])
    #     return data[0], data[1], cleanEqn, skeletonEqn, dataTest[0], dataTest[1], currEqn
    return cleanEqn, skeletonEqn, currEqn

######################################
# Use cases
######################################

# # Create a new random equation
# for i in range(10):
#     nv = 2
#     op_list= [
#                 "id", "add", "sub", "mul", "div",
#                 "sin", "cos", "exp", "log", "pow"
#             ]
#     numberofPoints = np.random.randint(30)
#     curr_eqn = generate_random_eqn_raw(n_vars=nv, op_list=op_list)
#     clean_eqn = eqn_to_str(curr_eqn, n_vars=nv)
#     print(clean_eqn)
#     print(eqn_to_str_skeleton(curr_eqn, n_vars=nv))
#     print()

# # Create data for that equation
# data = create_dataset_from_raw_eqn(curr_eqn, n_vars=nv, n_points=numberofPoints)
# print(data)

# print(dataGen(4, 3))
