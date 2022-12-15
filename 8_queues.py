import numpy as np
import collections
n_queues = 4
w = np.zeros([n_queues**2, n_queues**2])
theta = np.zeros(n_queues**2)
time_step = 100000


def energy_function(x_0):
    energy = 0
    for i in range(n_queues):
        x_ = 0
        x_2 = 0
        for j in range(n_queues):
            x_ += x_0[i][j]
            x_2 += x_0[j][i]
        energy += (x_-1)**2
        energy += (x_2 - 1) ** 2
    for i in range(1, n_queues-1):
        a = b = 0
        for j in range(i, n_queues):
            a += x_0[j-i][j]
            b += x_0[j][j-i]
        if a != 0:
            energy += (a - 1) ** 2
        if b != 0:
            energy += (b - 1) ** 2
    e = 0
    for j in range(0, n_queues):
        e += x_0[j][j]
    if e != 0:
        energy += (e-1)**2

    for i in range(1, n_queues-1):
        c = d = 0
        for k in range(0, n_queues-i):
            c += x_0[k][n_queues-1-k-i]
            d += x_0[k+i][n_queues-1-k]
        if c != 0:
            energy += (c - 1) ** 2
        if d != 0:
            energy += (d - 1) ** 2
    f = 0
    for j in range(0, n_queues):
        f += x_0[j][n_queues-1-j]
    if f != 0:
        energy += (f-1)**2
    return energy


c = energy_function(np.zeros([n_queues, n_queues]))
# get theta
for i in range(n_queues):
    for j in range(n_queues):
        mat = np.zeros([n_queues, n_queues])
        mat[i][j] = 1
        theta[i*n_queues + j] = energy_function(mat) - c

# get weights of the CNN network
for i in range(n_queues):
    for j in range(n_queues):
        for k in range(n_queues):
            for m in range(n_queues):
                mat = np.zeros([n_queues ** 2, n_queues ** 2])
                mat[i][j] = 1
                mat[k][m] = 1
                w[i*n_queues+j][k*n_queues+m] = -energy_function(mat) + \
                                                theta[i*n_queues+j] + theta[k*n_queues+m] + c
                if i*n_queues+j == k*n_queues+m:
                    w[i * n_queues + j][k * n_queues + m] = 0

print(f"weights: {w}; c: {c}, theta: {theta}")


def sigmoid(s_hat):
    return 1/(1+np.exp(-s_hat))


a = np.zeros([n_queues, n_queues])


def judge(x_0):
    for i in range(n_queues):
        for j in range(n_queues):

            if (x_0[i][j] == x_0).all():
                a[i][j] += 1


x = np.zeros(n_queues * n_queues)
# E = (2 * x[0] - x[1] + x[2] - 2) ** 2 + (-1 * x[0] + x[1] - x[2] + 1) ** 2 + (1 * x[0] - 2 * x[1] + x[2] - 1) ** 2
# E = -(w[0]*x[0]*x[1] + w[1]*x[1]*x[2] + w[2]*x[0]*x[2])
results = np.zeros([1, n_queues**2])

# deterministic model
for i in range(time_step):

    neurons = np.zeros([n_queues, n_queues])
    for m in range(n_queues):
        for n in range(n_queues):
            neurons[m][n] = x[m*n_queues + n]
    energy = energy_function(neurons)
    if energy == 0:
        results = np.vstack((results, x))
    print(f'value of x: {x}, energy: {energy}')

    # judge(x)
    for j in range(n_queues**2):
        s_hat = -theta[j]
        for k in range(n_queues**2):
            s_hat += w[j][k]*x[k]
        if s_hat >= 0:
            x[j] = 1
        else:
            x[j] = 0
#
dic = collections.Counter(map(tuple, results))

print(dic)



# # probabilistic model
# for i in range(time_step):
#
#     neurons = np.zeros([n_queues, n_queues])
#     for m in range(n_queues):
#         for n in range(n_queues):
#             neurons[m][n] = x[m*n_queues + n]
#     energy = energy_function(neurons)
#     if energy == 0:
#         results = np.vstack((results, x))
#     print(f'value of x: {x}, energy: {energy}')
#
#     # judge(x)
#     for j in range(n_queues**2):
#         s_hat = -theta[j]
#         for k in range(n_queues**2):
#             s_hat += w[j][k]*x[k]
#         p = sigmoid(s_hat)
#         if np.random.rand() < p:
#             x[j] = 1
#         else:
#             x[j] = 0
# #
# dic = collections.Counter(map(tuple, results))
#
# print(dic)

result1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) # 11
result2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 7
result3 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 7
result4 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # 7
result5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) # 7
result6 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) # 7
result7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # 7
result8 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) # 6
result9 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) # 6
result10 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # 6
result11 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # 6
result12 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 6
result13 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # 6
result14 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) # 5
result15 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # 5
result16 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) # 2506
result17 = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]) # 2479
# # plot
#
# from pygame_simulator import PyGame as PG
#
#
# def transfer_dim(mat):
#     res = np.zeros([n_queues, n_queues])
#     for i in range(n_queues):
#         for j in range(n_queues):
#             res[i][j] = mat[i*n_queues + j]
#     return res
#
#
# maps = np.zeros([n_queues, n_queues])
# pg = PG(n_queues, maps)
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# GREEN = (0, 255, 0)
# RED = (255, 0, 0)
# BLUE = (0, 0, 255)
# GRAY = (128, 128, 128)
# done_ = False
# while not done_:
#     for event in pg.event:
#         if event.type == pg.QUIT:
#             done_ = True
#
#     pg.screen.fill(BLACK)
#     pg.draw_map(n_queues - 1, n_queues - 1)
#     mat = transfer_dim(result17)
#
#     for i in range(n_queues):
#         for j in range(n_queues):
#             if mat[i][j] == 1:
#                 pg.draw_rect(i, j, RED)
#
#     # makes everything we've drawn visible all at once.
#     pg.display.flip()
#     # done_ = True


