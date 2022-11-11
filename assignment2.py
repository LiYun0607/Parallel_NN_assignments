import numpy as np
x = np.array([1, 0, 0])

theta = np.array([12, -10, 8])
w = np.array([10, 8, -8])
for i in range(2):
    for j in range(2):
        for k in range(2):
            x[0] = i
            x[1] = j
            x[2] = k
            E = (2 * x[0] - x[1] + x[2] - 2) ** 2 + (-1 * x[0] + x[1] - x[2] + 1) ** 2 + (1 * x[0] - 2*x[1] + x[2] - 1) ** 2
            print(f"E: {E}, x: {x}")
x = np.array([0, 0, 0])

for j in range(50):
    E = (2 * x[0] - x[1] + x[2] - 2) ** 2 + (-1 * x[0] + x[1] - x[2] + 1) ** 2 + (1 * x[0] - 2 * x[1] + x[2] - 1) ** 2
    print(f"x: {x}, E: {E}")
    # s = []
    # s.append(theta[0] + w[1] * x[1] + w[2] * x[2])
    # s.append(theta[1] + w[0] * x[0] + w[2] * x[2])
    # s.append(theta[2] + w[0] * x[0] + w[1] * x[2])
    s = theta[0] + w[1] * x[1] + w[2] * x[2]
    if s >= 0:
        x[0] = 1
        continue
    else:
        x[0] = 0
        continue
    s = theta[1] + w[0] * x[0] + w[2] * x[2]
    if s >= 0:
        x[1] = 1
        continue
    else:
        x[1] = 0
        continue
    s = theta[2] + w[1] * x[1] + w[0] * x[0]
    if s >= 0:
        x[2] = 1
        continue
    else:
        x[2] = 0
        continue

