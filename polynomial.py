import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from rnn_model import RNN


sns.set_theme(style='darkgrid')
log_dir = './data/'
os.makedirs(log_dir, exist_ok=True)
x = np.array([1, 1, 1])
x_ = []
E_ = []
time_step = 200000
n_neuron = 3

theta = np.array([-12, 10, -8])
w = np.zeros([3, 3])
sim_func = np.array([
    [2, -1, 1, 2],
    [-1, 1, -1, -1],
    [1, -2, 1, 1]
])


def energy_function(x_0):
    return (2*x_0[0]-x_0[1]+x_0[2]-2)**2 + (-1*x_0[0]+x_0[1]-x_0[2]+1)**2 + (1*x_0[0]-2*x_0[1]+x_0[2]-1)**2


c = energy_function([0, 0, 0])
theta[0] = energy_function([1, 0, 0]) - c
theta[1] = energy_function([0, 1, 0]) - c
theta[2] = energy_function([0, 0, 1]) - c
w[0][1] = w[1][0] = -energy_function([1, 1, 0]) + theta[0] + theta[1] + c
w[0][2] = w[2][0] = -energy_function([1, 0, 1]) + theta[0] + theta[2] + c
w[1][2] = w[2][1] = -energy_function([0, 1, 1]) + theta[2] + theta[1] + c
print(f"weights: {w}")

rnn = RNN(iteration=50, sim_func=sim_func, x0=x)
print(f"c, theta: {rnn.calc_func_para()}")
# for i in range(2):
#     for j in range(2):
#         for k in range(2):
#             x[0] = i
#             x[1] = j
#             x[2] = k
#             E = -(w[0]*x[0]*x[1] + w[1]*x[1]*x[2] + w[2]*x[0]*x[2])
#             E_.append(E)
#             x_.append(np.array([x]))
# data = pd.DataFrame({
#     "x": x_,
#     "energy": E_,
# })
# sns.lineplot(data=data)
# plt.show()
# log_dir_ = os.path.join(log_dir, 'data.csv')
# data.to_csv(log_dir_)


def sigmoid(s_hat):
    return 1/(1+np.exp(-s_hat))


a = np.zeros(2**3)
print(a)


def judge(x_0):
    if (x_0 == [0, 0, 0]).all():
        a[0] += 1.0
    if (x_0 == [1, 0, 0]).all():
        a[1] += 1.0
    if (x_0 == [0, 1, 0]).all():
        a[2] += 1.0
    if (x_0 == [0, 0, 1]).all():
        a[3] += 1.0
    if (x_0 == [1, 1, 0]).all():
        a[4] += 1.0
    if (x_0 == [1, 0, 1]).all():
        a[5] += 1.0
    if (x_0 == [0, 1, 1]).all():
        a[6] += 1.0
    if (x_0 == [1, 1, 1]).all():
        a[7] += 1.0


x = np.array([1, 1, 1])
E = (2 * x[0] - x[1] + x[2] - 2) ** 2 + (-1 * x[0] + x[1] - x[2] + 1) ** 2 + (1 * x[0] - 2 * x[1] + x[2] - 1) ** 2
# E = -(w[0]*x[0]*x[1] + w[1]*x[1]*x[2] + w[2]*x[0]*x[2])
print(f"x: {x}, E: {E}")

# deterministic model
for i in range(time_step):
    E = (2 * x[0] - x[1] + x[2] - 2) ** 2 + (-1 * x[0] + x[1] - x[2] + 1) ** 2 + (1 * x[0] - 2 * x[1] + x[2] - 1) ** 2
    print(f"value of x: {x}, E: {E}")
    judge(x)
    for j in range(n_neuron):
        s_hat = -theta[j]
        for k in range(n_neuron):
            s_hat += w[j][k]*x[k]
        if s_hat >= 0:
            x[j] = 1
        else:
            x[j] = 0


# probabilistic model
for i in range(time_step):
    print(f"value of x: {x}")
    judge(x)
    for j in range(n_neuron):
        s_hat = -theta[j]
        for k in range(n_neuron):
            s_hat += w[j][k]*x[k]
        p = sigmoid(s_hat)
        if np.random.rand() < p:
            x[j] = 1
        else:
            x[j] = 0

print(a)
if np.argmax(a) == 0:
    print("solution: 000")
if np.argmax(a) == 1:
    print("solution: 100")
if np.argmax(a) == 2:
    print("solution: 010")
if np.argmax(a) == 3:
    print("solution: 001")
if np.argmax(a) == 4:
    print("solution: 110")
if np.argmax(a) == 5:
    print("solution: 101")
if np.argmax(a) == 6:
    print("solution: 011")
if np.argmax(a) == 7:
    print("solution: 111")

