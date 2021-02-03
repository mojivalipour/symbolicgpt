import matplotlib.pyplot as plt
import numpy as np


# input_file = open("output_1var.txt", "r")
# input_lines = input_file.readlines()
# input_file.close()
#
# gp_errs = []
# mlp_errs = []
# sfl_errs = []
# num_less_than_0_1 = [0, 0, 0]
# num_less_than_0_01 = [0, 0, 0]
# num_less_than_0_5= [0, 0, 0]
# num_less_than_1= [0, 0, 0]
#
#
# new_input_lines = []
# for i in range(len(input_lines)-1):
#     if "Test case" in input_lines[i] and "Test case" in input_lines[i+1]:
#         continue
#     new_input_lines.append(input_lines[i].strip())
#
# # for line in new_input_lines:
# #     print(line)
#
# num_tests = int(len(new_input_lines)/7.0 + 0.5)
# print("{} tests".format(num_tests))
#
# for i in range(0, len(new_input_lines), 7):
#
#     eqn_index = int(new_input_lines[i].split("/")[0][10:].strip())
#     # print(eqn_index)
#     eqn_str = new_input_lines[i+1].strip()
#     # print(eqn_str)
#     mlp_err = min(np.exp(15), float(new_input_lines[i+2].split()[1]))
#     gp_err = min(np.exp(15), float(new_input_lines[i+4].split()[1]))
#     # mlp_err = float(new_input_lines[i + 2].split()[1])
#     # gp_err = float(new_input_lines[i + 4].split()[1])
#
#     gp_errs.append(gp_err)
#     mlp_errs.append(mlp_err)
#
#     if gp_err < 1:
#         num_less_than_1[0] += 1./num_tests
#         if gp_err < 0.5:
#             num_less_than_0_5[0] += 1./num_tests
#             if gp_err < 0.1:
#                 num_less_than_0_1[0] += 1./num_tests
#                 if gp_err < 0.01:
#                     num_less_than_0_01[0] += 1./num_tests
#
#     if mlp_err < 1:
#         num_less_than_1[1] += 1./num_tests
#         if mlp_err < 0.5:
#             num_less_than_0_5[1] += 1./num_tests
#             if mlp_err < 0.1:
#                 num_less_than_0_1[1] += 1./num_tests
#                 if mlp_err < 0.01:
#                     num_less_than_0_01[1] += 1./num_tests
#
#
# print("Number less than 1: {}".format(num_less_than_1))
# print("Number less than 0.5: {}".format(num_less_than_0_5))
# print("Number less than 0.1: {}".format(num_less_than_0_1))
# print("Number less than 0.01: {}".format(num_less_than_0_01))
#
# lists_of_error_scores = [gp_errs, mlp_errs]
#
# # print(lists_of_error_scores)
# model_names = ["GP", "MLP"]
#
#
# y1, x1, _ = plt.hist([np.log(errors_i) for errors_i in lists_of_error_scores],
#          label=[model_name for model_name in model_names],
#          cumulative=True, histtype="step", bins=num_tests, density="true")
#
#
# plt.subplot(121)
# plt.plot(x1[:-1], y1[0]*100, linestyle="-", label="GP")
# plt.plot(x1[:-1], y1[1]*100, linestyle="--", label="MLP")
#
#
# plt.legend(loc="upper left")
# plt.title("{} equations of {} variables".format(num_tests, 2))
# plt.xlabel("Log of error")
# plt.ylabel("Frequency")

###########################################

x = [_ for _ in range(6)]
y = [_ for _ in range(6)]
num_eqns = [0]

for num_vars in range(1, 6):

    input_file = open("output_{}var.txt".format(num_vars), "r")
    input_lines = input_file.readlines()
    input_file.close()

    gp_errs = []
    mlp_errs = []
    sfl_errs= []
    num_less_than_0_1 = [0, 0, 0]
    num_less_than_0_01 = [0, 0, 0]
    num_less_than_0_5= [0, 0, 0]
    num_less_than_1= [0, 0, 0]


    new_input_lines = []
    for i in range(len(input_lines)-1):
        if "Test case" in input_lines[i] and "Test case" in input_lines[i+1]:
            continue
        new_input_lines.append(input_lines[i].strip())

    num_tests2 = int(len(new_input_lines)/7.0 + 0.5)
    print("{} tests".format(num_tests2))


    for i in range(0, len(new_input_lines), 7):
        # for line in new_input_lines[i:i+7]:
        #     print(line)

        eqn_index = int(new_input_lines[i].split("/")[0][10:].strip())
        eqn_str = new_input_lines[i+1].strip()
        mlp_err = min(np.exp(15), float(new_input_lines[i+2].split()[1]))
        gp_err = min(np.exp(15), float(new_input_lines[i+4].split()[1]))

        gp_errs.append(gp_err)
        mlp_errs.append(mlp_err)

        if gp_err < 1:
            num_less_than_1[0] += 1./num_tests2
            if gp_err < 0.5:
                num_less_than_0_5[0] += 1./num_tests2
                if gp_err < 0.1:
                    num_less_than_0_1[0] += 1./num_tests2
                    if gp_err < 0.01:
                        num_less_than_0_01[0] += 1./num_tests2

        if mlp_err < 1:
            num_less_than_1[1] += 1./num_tests2
            if mlp_err < 0.5:
                num_less_than_0_5[1] += 1./num_tests2
                if mlp_err < 0.1:
                    num_less_than_0_1[1] += 1./num_tests2
                    if mlp_err < 0.01:
                        num_less_than_0_01[1] += 1./num_tests2


    print("Number less than 1: {}".format(num_less_than_1))
    print("Number less than 0.5: {}".format(num_less_than_0_5))
    print("Number less than 0.1: {}".format(num_less_than_0_1))
    print("Number less than 0.01: {}".format(num_less_than_0_01))

    lists_of_error_scores = [gp_errs, mlp_errs]
    model_names = ["GP", "MLP"]

    y[num_vars], x[num_vars], _ = plt.hist([np.log(errors_i) for errors_i in lists_of_error_scores],
                                  label=[model_name for model_name in model_names],
                                  cumulative=True, histtype="step", bins=num_tests2, density="true")

    num_eqns.append(num_tests2)


plt.figure(figsize=(15, 10))
for num_vars in range(1, 6):
    plt.subplot(2, 3, num_vars)
    plt.plot(x[num_vars][:-1], y[num_vars][0]*100, linestyle="-", label="GP")
    plt.plot(x[num_vars][:-1], y[num_vars][1]*100, linestyle="--", label="MLP")


    plt.legend(loc="upper left")
    plt.title("{} equations of {} variables".format(num_eqns[num_vars], num_vars))
    plt.xlabel("Log of error")
    plt.ylabel("Frequency")






plt.savefig("images/hist_of_errors.png")
plt.close()

