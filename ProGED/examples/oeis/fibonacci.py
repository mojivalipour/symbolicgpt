import numpy as np
from ProGED.equation_discoverer import EqDisco

oeis = [0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
 1597,2584,4181,6765,10946,17711,28657,46368,75025,
 121393,196418,317811,514229,832040,1346269,
 2178309,3524578,5702887,9227465,14930352,24157817,
 39088169,63245986,102334155]
# oeis_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
#  61,67,71,73,79,83,89,97,101,103,107,109,113,127,
#  131,137,139,149,151,157,163,167,173,179,181,191,
#  193,197,199,211,223,227,229,233,239,241,251,257,
#  263,269,271][:40+1]
# oeis = oeis_primes
fibs = np.array(oeis).reshape(-1, 1)
ts = np.array([i for i in range(40+1)]).reshape(-1, 1)
data = np.hstack((ts, fibs))

np.random.seed(0)
ED = EqDisco(data = data,
            task = None,
            target_variable_index = -1,
            variable_names=["n", "an"],
            sample_size = 10,
            verbosity = 0,
            generator = "grammar", 
            generator_template_name = "polynomial",
            generator_settings={"variables":["'n'"]},
            estimation_settings={"verbosity": 0, "task_type": "algebraic", "lower_upper_bounds": 
            # (-5, 5)} 
            # (-10,8)} # Last bound where it still finds.
            (0,1)}  # Returns wrong error on windows.
            )
ED.generate_models()
ED.fit_models()
# print(ED.models)
print("\n\nFinal score:")
for m in ED.models:
    print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    
phi = (1+5**(1/2))/2
psi = (1-5**(1/2))/2
c0 = 1/5**(1/2)
c1 = np.log(phi)
print(f"m  c0: {c0}", f"c1:{c1}")
# fib(n) = (phi**n - psi**n)/5**(1/2)
#         = round(phi**n/5**(1/2))
#         = floor(phi**n/5**(1/2) + 1/2)

# model = ED.models[5] 
# # model = ED.models[15] 
# print(model, model.get_full_expr(), model.get_error())
# res = model.evaluate(ts, *model.params)
# res = [int(np.round(flo)) for flo in res]

# print(res)
# print(oeis)
# error = 0
# for i, j in zip(res, oeis):
#     print(i,j, i-j, error)
#     error += abs(i-j)

# print(error)

