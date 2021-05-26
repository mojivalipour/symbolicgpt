import numpy as np
# import sympy as sp
# # pg = ProGED
# import ProGED as pg
# # from pg.generate import generate_models
# # gft = pg.generators.grammar_from_template
# # from ProGED import generators.generate_models
from ProGED.equation_discoverer import EqDisco
# # from ProGED.generators.grammar import GeneratorGrammar
# from ProGED.generators.grammar_construction import grammar_from_template
# from ProGED.generate import generate_models
# # from ProGED.model import Model
# # from ProGED.model_box import ModelBox
# # from ProGED.parameter_estimation import fit_models

oeis = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
 1597,2584,4181,6765,10946,17711,28657,46368,75025,
 121393,196418,317811,514229,832040,1346269,
 2178309,3524578,5702887,9227465,14930352,24157817,
 39088169,63245986,102334155]
fibs = np.array(oeis)
# ts = np.array([i for i in range(40+1)]).reshape(-1, 1)
# ts = np.array([i for i in range(40+1)])
# print(ts, type(fibs), fibs.shape, type(ts[0]), ts.shape)
# data = np.hstack((ts, fibs))

# map( [0..m], fibs[i:-m+i])

# map([1,2,])

# we want:
# 0 1 2
# 1 2 3
# 2 3 4
# m = 2
# (n-m, m)
# def f(i,j):
#    return i+j 
# template = np.fromfunction((lambda i,j:i+j), (40+1-2,2+1), dtype=int)

def mdata(m, fibs):
    """m -> (n-m) x (0/1+1+m) data matrix
    m ... number of previous elements in formula
    """
    n = fibs.shape[0] # (40+1)
    # if ts.shape != fibs.shape:
    #     print("ts and fibs differend dimensions !!!!!")
    #     return 1/0
    # np.hstack(ts[:-m].reshape(-1,1), fibs[:-m])
    indexes = np.fromfunction((lambda i,j:i+j), (n-m, m+1), dtype=int)
    # first_column = indexes[:, [0]]
    # return np.hstack((first_column, fibs[indexes]))
    return fibs[indexes]
# print(fibs)
# print(mdata(2, fibs))
data = mdata(2, fibs)




# # variables = ["'n'"]
# # symbols = {"x":variables, "start":"S", "const":"C"}
# # # p_vars = [1, 0.3, 0.4]
p_T = [0.4, 0.6]
p_R = [0.9, 0.1]
# # grammar = grammar_from_template("polynomial", {"variables": variables, "p_R": p_R, "p_T": p_T})
# # np.random.seed(0)
# # # print(grammar.generate_one())
# # # models = generate_models(grammar, symbols, strategy_settings = {"N":500})
# # # print(models)

np.random.seed(0)
# seed 0 , size 20 (16)
# seed3 size 15 an-1 + an-2 + c3
ED = EqDisco(data = data,
            task = None,
            target_variable_index = -1,
            variable_names=["an_2", "an_1", "an"],
            sample_size = 16,
            verbosity = 0,
            generator = "grammar", 
            generator_template_name = "polynomial",
            generator_settings={"variables": ["'an_2'", "'an_1'"], "p_T": p_T, "p_R": p_R}
            ,estimation_settings={"verbosity": 1, "task_type": "algebraic", "lower_upper_bounds": 
            (0, 2)} # meja, ko se najde priblizno: (-10,8)}# , "timeout": np.inf}
            )
# # print(data, data.shape)
ED.generate_models()
ED.fit_models()
# # try:
# #     print(12/0)

print(ED.models)
# # print(ED.get_results())
# # print(ED.get_stats())
# print("\nFinal score:")
# for m in ED.models:
#     print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    
# phi = (1+5**(1/2))/2
# psi = (1-5**(1/2))/2
# c0 = 1/5**(1/2)
# c1 = np.log(phi)
# print(f"m  c0: {c0}", f"c1:{c1}")
# # fib(n) = (phi**n - psi**n)/5**(1/2)
# #         = round(phi**n/5**(1/2))
# #         = floor(phi**n/5**(1/2) + 1/2)

# # model = ED.models[5] 
model = ED.models[-1] 
# res = model.evaluate(ts, *model.params)
# res = [int(np.round(flo)) for flo in res]

# print(res)
# print(oeis)
# error = 0
# for i, j in zip(res, oeis):
#     print(i,j, i-j, error)
#     error += abs(i-j)

# print(error)

