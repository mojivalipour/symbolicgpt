"""Run equation discovery on OEIS sequences to discover 
direct, recursive or even direct-recursive equations.
"""

import numpy as np
import pandas as pd
import time
import sys
# from scipy.optimize import brute, shgo, rosen, dual_annealing

from ProGED.equation_discoverer import EqDisco
from ProGED.parameter_estimation import DE_fit, hyperopt_fit  # integer_brute_fit, shgo_fit, DAnnealing_fit
import ProGED.examples.tee_so as te  # Log using manually copied class from a forum.
from hyperopt import hp, tpe, rand


#####  To log output to file, for later inspection.  ########
# 9.53 - 20min
# Command line arguments (to avoid logging):
is_tee_flag = True  # Do not change manually!! Change is_tee.
message = ""
double_flags = set(sys.argv[1:])
flags_dict = { i.split("=")[0]:  i.split("=")[1] for i in sys.argv[1:] if len(i.split("="))>1}
# Usage of flags_dict: $ fibonacci.py --order=3 --is_direct=True.
# List of all flags:
#   -n or --no-log .. do not log  # --do-log .. do log  # --msg .. log name  
# --is_direct=<bool>  # --order=<num>  # --sample_size=<num>
if len(sys.argv) >= 2:
    if sys.argv[1][0] == "-" and not sys.argv[1][1] == "-":
        single_flags = set(sys.argv[1][1:])
        # print(single_flags)
        if "n" in single_flags:
            is_tee_flag = False
    if "--no-log" in double_flags:
        is_tee_flag = False
    if "--msg" in double_flags:
        message = sys.argv[2] + "_"
    if "--is_direct" in flags_dict:
        if flags_dict["--is_direct"] == "True":
            flags_dict["--is_direct"] = True
        elif flags_dict["--is_direct"] == "False":
            flags_dict["--is_direct"] = False
        else:
            flags_dict["--is_direct"] = int(flags_dict["--is_direct"])
if not is_tee_flag:
    print("\nNo-log flag detected!\n")

is_tee, log_name, log_directory = False, "log_oeis_", "outputs/"
# is_tee, log_name, log_directory = True, "log_oeis_", "outputs/"
random = str(np.random.random())[2:]
log_filename = log_name + message + random + ".txt"
if (is_tee and is_tee_flag) or ("--do-log" in double_flags):
    try:
        log_object = te.Tee(f"{log_directory}{log_filename}")
        print(f"Output will be logged in: {log_directory}{log_filename}")
    except FileNotFoundError:
        log_object = te.Tee(log_filename)
        print(f"Output will be logged in: {log_filename}")
######## End Of Logging ##  ##  ##                  ########

np.random.seed(0)
has_titles = 1
csv = pd.read_csv('oeis_selection.csv')[has_titles:]
# csv = csv.astype('int64')
csv = csv.astype('float')
terms_count, seqs_count = csv.shape
# Old for fibonacci only:
# seq_id = "A000045"
# fibs = list(csv[seq_id])  # fibonacci = A000045
# oeis = fibs

# oeis = fibs[2:]
# fibs_old = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
#  1597,2584,4181,6765,10946,17711,28657,46368,75025,
#  121393,196418,317811,514229,832040,1346269,
#  2178309,3524578,5702887,9227465,14930352,24157817,
#  39088169,63245986,102334155]
# print(type(fibs[0]))
# assert fibs_old[len(fibs_old)-1] == fibs[len(fibs_old)-1]
# primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
#  61,67,71,73,79,83,89,97,101,103,107,109,113,127,
#  131,137,139,149,151,157,163,167,173,179,181,191,
#  193,197,199,211,223,227,229,233,239,241,251,257,
#  263,269,271][:40+1]
# oeis = primes
# seq = numpy_seq

# seq = np.array(oeis)

# we want:
# 0 1 2
# 1 2 3
# 2 3 4
# an = a{n-1} + a{n-2} is recurrece relation of order 2 (2 preceeding terms).
def grid (order, seq, direct=False):
    """order -> (n-order) x (0/1+order+1) data matrix
    order ... number of previous terms in formula
    0/1+ ... for direct formula also (a_n = a_{n-1} + n).
    +1 ... a_n column.
    """
    n = seq.shape[0] # (40+1)
    indexes = np.fromfunction((lambda i,j:i+j), (n-order, order+1), dtype=int)
    first_column = indexes[:, [0]]
    if direct:
        return np.hstack((first_column, seq[indexes]))
    return seq[indexes]

#######main#settings#####################################
# Note: order and is_direct is overwritten by commandline arguments.
order, is_direct = 2, False  # recursive
# order, is_direct = 4, False  # recursive
# order, is_direct = 0, True  # direct
# Override manual settings with input cmd line flags:
order = int(flags_dict.get("--order", order))
is_direct = flags_dict.get("--is_direct", is_direct)

# seq_name = "fibonacci"
seq_name = "general_wnb"
grammar_template_name = "polynomial"
# sample_size = 1
sample_size = 4
# sample_size = 2
# sample_size = 3
# sample_size = 6
# sample_size = 16
# sample_size = 15
# sample_size = 20
# sample_size = 20
# sample_size = 100
sample_size = int(flags_dict.get("--sample_size", sample_size))
lower_upper_bounds = (-5, 5) if is_direct else (-10, 10)
# lower_upper_bounds = (-10, 10)  # recursive
# lower_upper_bounds = (-5, 5)  # direct
# lower_upper_bounds = (-2, 2)  # direct
#########################################################

p_T = [0.4, 0.6]  # default settings, does nothing
p_R = [0.6, 0.4]
# if (seq_name, order, is_direct) == ("fibonacci", 2, False):
#     p_T = [0.4, 0.6]  # for rec fib
#     p_R = [0.9, 0.1]

# random_seed = 0
random_seed = 1  # rec
# seed 0 , size 20 (16)
# ??? seed3 size 15 an-1 + an-2 + c3 rec  ???
# seed 1 size 20 ali 4 an-1 + an-2 rec 

task_type = "algebraic"  # Originalno blo nastimano do 5.5.2021.
# task_type = "integer-algebraic"  # Originalno blo nastimano do 5.5.2021.
optimizer = 'differential_evolution'
timeout = np.inf

def oeis_eq_disco(seq_id: str, is_direct: bool, order: int): 
    """Run eq. discovery of given OEIS sequence."""

    data = grid(order, np.array(list(csv[seq_id])), is_direct)
    variable_names_ = [f"an_{i}" for i in range(order, 0, -1)] + ["an"]
    variable_names = ["n"]+variable_names_ if is_direct else variable_names_
    variables = [f"'{an_i}'" for an_i in variable_names[:-1]]
    # print(variable_names)
    # print(variables)
    # print(data.shape, type(data), data)

    np.random.seed(random_seed)
    ED = EqDisco(
        data = data,
        task = None,
        target_variable_index = -1,
        # variable_names=["an_2", "an_1", "an"],
        variable_names=variable_names,
        # sample_size = 16,  # for recursive
        # sample_size = 10,  # for direct fib
        sample_size = sample_size,
        # sample_size = 50,  # for recursive
        # sample_size = 38,  # for recursive
        # sample_size = 100,  # for recursive
        verbosity = 0,
        # verbosity = 3,
        generator = "grammar", 
        generator_template_name = grammar_template_name,
        # generator_settings={"variables": ["'an_2'", "'an_1'"],
        generator_settings={"variables": variables,
                             "p_T": p_T, "p_R": p_R
                             },

        estimation_settings={
            # "verbosity": 3,
            "verbosity": 1,
            # "verbosity": 0,
             "task_type": task_type,
             # "task_type": "algebraic",
             # "task_type": "integer-algebraic",
            # "task_type": "oeis_recursive_error",  # bad idea
             "lower_upper_bounds": lower_upper_bounds,
             # "lower_upper_bounds": (-1000, 1000),  # najde (1001 pa ne) rec fib
            #  "lower_upper_bounds": (-100, 100),  # ne najde DE
            #  "lower_upper_bounds": (-25, 25),  # DA ne najde
            #  "lower_upper_bounds": (-11, 11),  # shgo limit
            #  "lower_upper_bounds": (-14, 14),  # DA dela
            # "lower_upper_bounds": (-2, 2),  # for direct
            # "lower_upper_bounds": (-4, 4),  # for direct fib
            # "lower_upper_bounds": (-5, 5), 
            # "lower_upper_bounds": (-8, 8),  # long enough for brute
            # "optimizer": 'differential_evolution',
            "optimizer": optimizer,
            # "optimizer": 'hyperopt',
            "timeout": timeout,
            # "timeout": 1,
            # "timeout": 13,
            # "timeout_privilege": 30,

                ## hyperopt:
            # "hyperopt_max_evals": 3250,
            # "hyperopt_max_evals": 550,  # finds if result=min(10**6, hyperopt...)
            # "hyperopt_max_evals": 50,
            # "hyperopt_max_evals": 750,
            # "hyperopt_max_evals": 2000,
            # "hyperopt_max_evals": 700,
            # "hyperopt_max_evals": 500,
            # "hyperopt_max_evals": 300,
            # "hyperopt_space_fn": hp.randint,
            # "hyperopt_space_fn": hp.choice,
            # "hyperopt_space_fn": hp.loguniform,  # Seems working, but loss 785963174750.8921 in 2000 evals.
            # "hyperopt_space_fn": hp.qnormal,
            # "hyperopt_space_fn": hp.normal,
            # "hyperopt_space_args": (lower_upper_bounds[0], lower_upper_bounds[1]),
            # "hyperopt_space_args": ([-2, -1, 0, 1],),
            # "hyperopt_space_args": (0, 5, 1),
            # "hyperopt_space_args": (0, 2),  # Seems working for hp.normal, but loss 1492702199922.7131 in 2000 evals.
            # "hyperopt_space_args": [lower_upper_bounds[0]],
            # "hyperopt_space_kwargs": {"high": lower_upper_bounds[1]},
            # "hyperopt_algo": tpe.suggest,  # Works, with 1000 evals and (-2, 2) finds recursive. 
            # "hyperopt_algo": rand.suggest,
        }
    )


    # for i in range(0, 100):
    #     np.random.seed(i)
    #     print("seed", i)
    #     print(ED)
    #     ED.generate_models()
    #     print(ED.models)
    #     ED.models = None

    # 1/0
    ED.generate_models()
    print(ED.models)
    # 1/0
    seq_start = time.perf_counter()
    ED.fit_models()
    seq_cpu_time = time.perf_counter() - seq_start
    print(f"\nParameter fitting for sequence {seq_id} "
          f"took {seq_cpu_time} secconds.")
    print("\nFinal score:")
    for m in ED.models:
        print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    return

# oeis_eq_disco(seq_id, is_direct, order)  # Run only one seq, e.g. the fibonaccis.
# oeis_eq_disco("A000045", is_direct, order)  # Run only one seq, e.g. the fibonaccis.
# Run eq. disco. on all oeis sequences:
print("Running equation discovery for all oeis sequences, "
        "with these settings:\n"
        f"=>> is_direct = {is_direct}\n"
        f"=>> order of equation recursion = {order}\n"
        f"=>> sample_size = {sample_size}\n"
        f"=>> grammar_template_name = {grammar_template_name}\n"
        f"=>> generator_settings = {'{'}p_T: {p_T}, p_R: {p_R}{'}'}\n"
        f"=>> optimizer = {optimizer}\n"
        f"=>> task_type = {task_type}\n"
        f"=>> timeout = {timeout}\n"
        f"=>> number of terms in every sequence = {terms_count}\n"
        f"=>> number of all considered sequences = {seqs_count}\n"
        f"=>> list of considered sequences = {list(csv.columns)}\n"
    )
start = time.perf_counter()
FIRST_ID = "A000000"
LAST_ID = "A246655"
# last_run = "A002378"

start_id = FIRST_ID
start_id = "A000045"
end_id = LAST_ID
end_id = "A000045"
csv = csv.loc[:, (start_id <= csv.columns) & (csv.columns <= end_id)]
for seq_id in csv:
    oeis_eq_disco(seq_id, is_direct, order)
    print(f"\nTotal time consumed by now:{time.perf_counter()-start}\n")
cpu_time = time.perf_counter() - start
print(f"\nEquation discovery for all (chosen) OEIS sequences"
      f" took {cpu_time} secconds, i.e. {cpu_time/60} minutes"
      f" or {cpu_time/3600} hours.")

def pretty_results(seq_name="fibonacci", is_direct=is_direct, order=order):
    """Print results in prettier form."""
    if seq_name =="fibonacci":
        assert oeis == fibs
    if seq_name=="fibonacci" and is_direct and order==0:  # i.e. direct fib
        # is_fibonacci_direct = True
        # if is_fibonacci_direct:
        phi = (1+5**(1/2))/2
        c0, c1 = 1/5**(1/2), np.log(phi)
        print(f" m c0: {c0}", f"c1:{c1}")
        model = ED.models[5]  # direct fib
    elif seq_name=="fibonacci" and not is_direct and order != 0:  # i.e. rec fib
        model = ED.models[-1]
    elif seq_name=="primes" and not is_direct and order != 0:  # i.e. rec primes
        model = ED.models[7]  # primes
    else:    
        model = ED.models[-1]  # in general to update
        
    # an = model.lambdify()
    an = model.lambdify(*np.round(model.params)) if order != 0 else model.lambdify(*model.params)
    print("model:", model.get_full_expr())#, "[f(1), f(2), f(3)]:", an(1), an(2), an(3))

    cache = oeis[:order]  # update this
    # cache = list(oeis[:order])
    for n in range(order, len(oeis)):
        prepend = [n] if is_direct else []
        append = cache[-order:] if (order != 0) else []
        cache += [int(np.round(an(*(prepend+append))))]
        # print(prepend, append, prepend + append, (prepend + append), cache, an)
    res = cache
    print(oeis)
    print(res)
    error = 0
    for i, j in zip(res, oeis):
        print(i,j, i-j, error)
        error += abs(i-j)
    print(error)
    return
# pretty_results(seq_name=seq_name, is_direct=is_direct, order=order)

