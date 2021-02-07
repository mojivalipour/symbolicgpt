import matplotlib.pyplot as plt
import numpy as np

num_to_plot = 1

###########################################

x = [_ for _ in range(num_to_plot+1)]
y = [_ for _ in range(num_to_plot+1)]
num_eqns = [0]

for num_vars in range(1, num_to_plot+1):

    input_file = open("output_old{}var.txt".format(num_vars), "r")
    input_lines = input_file.readlines()
    input_file.close()

    gp_errs = []
    mlp_errs = []
    gpt_errs= []
    ptd_errs = []
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
        for line in new_input_lines[i:i+7]:
            print(line)

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

    input_file = open("gpt_output_{}var.txt".format(num_vars), "r")
    input_lines = input_file.readlines()
    input_file.close()

    for i in range(0, len(input_lines), 5):
        eqn_index = int(input_lines[i].split("/")[0][10:].strip())
        eqn_str = input_lines[i+1].strip()
        raw_err = input_lines[i+2].split()[1]
        if "Not" in raw_err:
            continue
        raw_err = float(raw_err)
        gpt_err = min(np.exp(15), np.exp(raw_err))

        gpt_errs.append(gpt_err)

        if gpt_err < 1:
            num_less_than_1[2] += 1./num_tests2
            if gpt_err < 0.5:
                num_less_than_0_5[2] += 1./num_tests2
                if gpt_err < 0.1:
                    num_less_than_0_1[2] += 1./num_tests2
                    if gpt_err < 0.01:
                        num_less_than_0_01[2] += 1./num_tests2


    print("Number less than 1: {}".format(num_less_than_1))
    print("Number less than 0.5: {}".format(num_less_than_0_5))
    print("Number less than 0.1: {}".format(num_less_than_0_1))
    print("Number less than 0.01: {}".format(num_less_than_0_01))

    if num_vars in [1, 3]:
        input_file = open("ptdeep_output_old{}var.txt".format(num_vars), "r")
        input_lines = input_file.readlines()
        input_file.close()

        for i in range(0, len(input_lines), 5):
            eqn_index = int(input_lines[i].split("/")[0][10:].strip())
            eqn_str = input_lines[i + 1].strip()
            raw_err = input_lines[i + 2].split()[1]
            if "Not" in raw_err:
                continue
            raw_err = float(raw_err)
            ptdeep_err = min(np.exp(15), np.exp(raw_err))

            ptd_errs.append(ptdeep_err)


        lists_of_error_scores = [gp_errs, mlp_errs, gpt_errs, ptd_errs]
        model_names = ["GP", "MLP", "GPT", "PT-Deep"]
    else:
        lists_of_error_scores = [gp_errs, mlp_errs, gpt_errs]
        model_names = ["GP", "MLP", "GPT"]

    y[num_vars], x[num_vars], _ = plt.hist([np.log(errors_i) for errors_i in lists_of_error_scores],
                                  label=[model_name for model_name in model_names],
                                  cumulative=True, histtype="step", bins=num_tests2, density="true")

    num_eqns.append(num_tests2)


plt.figure(figsize=(15, 10))
for num_vars in range(1, num_to_plot+1):
    plt.subplot(min(2, int((num_to_plot+1)/2)), int((num_to_plot+1)/2), num_vars)
    plt.plot(x[num_vars][:-1], y[num_vars][2] * 100, linestyle="-", label="GPT")
    plt.plot(x[num_vars][:-1], y[num_vars][0]*100, linestyle="dashdot", label="GP")
    plt.plot(x[num_vars][:-1], y[num_vars][1]*100, linestyle="dotted", label="MLP")
    if num_vars in [1, 3]:
        plt.plot(x[num_vars][:-1], y[num_vars][3] * 100, linestyle="--", label="PT-Deep")

    plt.legend(loc="upper left")
    plt.title("{} equations of {} variables".format(num_eqns[num_vars], num_vars))
    plt.xlabel("Log of error")
    plt.ylabel("Frequency")






plt.savefig("images/hist_of_errors.png")
plt.close()

