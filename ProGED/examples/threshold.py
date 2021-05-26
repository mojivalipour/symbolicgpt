"""Filters results, i.e. output of equation discovery from lorenz.py
in output files, to calculate % of equations better than the right 
equation based on their error and threshold.
"""

import re
import sys

from ProGED.examples.tee_this import create_log

# Command line arguments:
file_name = sys.argv[1] 
equation_name = sys.argv[2]

create_log("threshold-" + file_name, with_random=False) 
with open(file_name, "r") as file:
    string = file.read()
model_box = re.findall("ModelBox: (\d+) models", string)
fitted_models = str(model_box[0])
print(fitted_models, "Number of fitted models")

# Crop only the final results out of the output file.
print_models = re.findall("Final score:\n[^w]*", string)

# Relevant threshold for each equation discovery, i.e. error of 
# data generating model to compare against.
equation_dict = {
"dx_calm" : """
%  model: -1.3009139792956*x + 1.29951581073809*y 
 % p: 0.0015054336000000002  ; error: 2.5822654126803468e-11
""",
"dx_atrac" : """
% model: -9.92174242687289*x + 9.96421639351077*y
%; p: 0.0015054336000000002; error: 1.641181498418429e-08
""",
"dy_calm" : """
model: -0.834781594777948*x*z - 15.0440585292989*x - 1.01069280464677*y      ; p: 6.50962591625742e-09   ; error: 2.05234552266566e-09
""",
"dy_atrac" : """
% model: -0.997865321264281*x*z + 28.0869577139579*x - 1.04038150375494*y      
% ; p: 6.50962591625742e-09; error: 7.250727384603807e-08
""",
"dz_calm" : """
% model: 1.00251669434065*x*y - 3.40016786784472*z
% ; p: 0.00020473896960000005 ; error: 2.2642219744753384e-10
""",
"dz_atrac" : """
% model: 1.00033849472896*x*y - 2.67305013976617*z                             
% ; p: 0.00020473896960000005; error: 1.1976285788575919e-08
"""}
key_line = equation_dict[equation_name]  # = e.g. dx_calm
print("Key line, i.e. line of most interest in output of Eq Disco:", key_line)
# Obtain the threshold:
error_per_partes = re.findall("error:.(([\d\.]+)(e(\-\d\d))*)", key_line)
if not len(error_per_partes):
    print("ERROR IN KEYLINE!!!")
error_str, coef, _, exp = error_per_partes[0]  # = 2.5822654126803468, e-11, -11
error = float(error_str)
threshold = error

# Do threshold filter via the key_line (our watched equation):
lines = re.findall("model:.*error:.[\d\.e\-]+", print_models[0])
filtered_lines = [line for line in lines if float(re.findall("error:.([\d\.e\-]+)", line)[0]) <= threshold]
if len(filtered_lines) == 0:
    print("ERROR IN THRESHOLD!!!")
elif len(filtered_lines) == 1:
    n = 3
    print(f"This was the best eq. found. Therefore lets print {n} (other) equations with very small error.")
    # of slightly bigger order (+1)")
    stop = False
    for i in range(10):
        filtered_lines_th2 = [line for line in lines 
                if float(re.findall("error:.([\d\.e\-]+)", line)[0]) <= threshold*10**i]
        if len(filtered_lines_th2) >= n: stop = True
        if stop: break
    if not stop: print("Only our model! -> something went wrong!!")
    for i in filtered_lines_th2[:n]:
        print(i)
    
print("\nFiltered lines: ")
for i in filtered_lines:
    print(i)
print("\n",len(lines), "regexed lines")
print(len(filtered_lines), "filtered lines")
print(fitted_models, "Number of fitted models")

# output of this program:
percent_raw = len(filtered_lines)/len(lines)*100
percent = round(percent_raw, 2)
print(percent_raw)
print(f"The original equation was found in the top {percent}% of estimated models.")
# results:
# dx calm: 7.14% - 2st of 100 
# dy calm: 0.52% 
# dz calm: 2.44% - 1st of 100 
# dx atrac: 7.14% - 2st of 100 
# dy atrac: 0.17% 
# dz atrac: 2.44% - 1st of 100 
