import torch
import csv
import matplotlib.pyplot as plt
import numpy as np

PRINT = "time(s)"
PRINT = "epoch"
n = 500
m = 500
epoch = 300
time = 50

methods = [
    ["GD", "GD", 0.5, 1.0, 1],
    ["GD", "Newton", 1.0, 1.0, 1],
    ["Newton", "Newton", 1.0, 1.0, 1],
    ["GD", "BFGS", 1.0, 0.5, 3],
    ["BFGS", "BFGS", 0.5, 0.5, 2],
    ["GD", "LBFGS", 1.0, 0.5, 3, 3],
    ["LBFGS", "LBFGS", 0.5, 0.5, 2, 10]
]
methods_name = [
    "GDA",
    "GDN",
    "CN",
    "GDBFGS",
    "CBFGS",
    "GDLBFGS",
    "CLBFGS"
]
methods_color = [
    "blue",
    "magenta",
    "green",
    "orange",
    "red",
    "purple",
    "gray"
]
methods_linestyle = [
    "--",
    "--",
    "--",
    "-",
    "-",
    "-",
    "-"
]
num_methods = len(methods)

def get_lst(opt_x, opt_y, lr_x, lr_y, num_y, mem_y = 0):
    
    number_list = []
    pattern = "./SCSC/{}-{}/{}-{}-{}-{}-{}/"
    with open(pattern.format(n, m, opt_x, opt_y, lr_x, lr_y, num_y) + "time.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each item in the row to a float and add to the list
            number_list.append([float(item) for item in row])
        x, y, t = [list(row) for row in zip(*number_list)]
        if PRINT == "time(s)":
            for i in range(len(t)):
                if t[i] > time:
                    return [x[:i], y[:i], t[:i]]
            return [x, y, t]
        elif PRINT == "epoch":
            return [x[:epoch], y[:epoch], np.arange(0, len(x[:epoch]), 1)]
lst_m = []
for i in range(num_methods):
    s = get_lst(*methods[i])
    lst_m.append(s)

fig, axes = plt.subplots(figsize=(7.5, 3.5), nrows=1, ncols=2)
for i in range(num_methods):
    axes[0].plot(lst_m[i][2], [np.linalg.norm(xx) for xx in lst_m[i][0]], linewidth=3, linestyle=methods_linestyle[i], label=methods_name[i], color=methods_color[i])
axes[0].set_yscale('log')
axes[0].set_xlabel(PRINT, fontsize=30)
axes[0].set_ylabel(r" $\vert| x - x^* \vert|$", fontsize=30)
axes[0].tick_params(labelsize=30)
axes[0].legend(loc='upper right', fontsize=25)
for i in range(num_methods):
    axes[1].plot(lst_m[i][2], [np.linalg.norm(xx) for xx in lst_m[i][1]], linewidth=3, linestyle=methods_linestyle[i], label=methods_name[i], color=methods_color[i])
axes[1].set_yscale('log')
axes[1].set_xlabel(PRINT, fontsize=30)
axes[1].set_ylabel(r"$\vert| y - y^* \vert|$", fontsize=30)
axes[1].tick_params(labelsize=30)
axes[1].legend(loc='upper right', fontsize=25)

fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
plt.show()