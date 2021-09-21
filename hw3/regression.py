import numpy as np

def train_gd(train_data, labels, steps = 10, rate = 0.001):

    W = 3
    b = 0
    N = train_data.shape[0]

    for i in range(steps):

        J = np.sum((W * train_data + b - labels) ** 2) / N
        dw = 0
        db = 0

        for j in range(N):
            common = W * train_data[j] + b - labels[j]
            db += common
            dw += (common * train_data[j])

        db = db * 2. / N
        dw = dw * 2. / N

        W = W - rate * dw
        b = b - rate * db

    if i + 1 == steps:

        print(f'{i = }')
        print(f'{J = }')
        print(f'{dw = }')
        print(f'{db = }')
        print(f'{W = }')
        print(f'{b = }\n')


X = np.array([9.0, 2, 6, 1, 8])
Y = np.array([1.0, 0, 3, 0, 1])

train_gd(X, Y, 5, 0.001)