import numpy as np

def train(train_data, labels):

    # Adding 1 corresponding to b.
    x_final = np.ones(train_data.shape[0]).reshape(-1, 1)
    train_data = np.hstack((train_data, x_final))
    theta = np.zeros((train_data.shape[1])).reshape(-1, 1)

    print(f'{theta.shape = }')
    print(f'{train_data.shape = }')
    print(train_data)
    print(labels)

    isConverged = False
    k = 0

    while isConverged == False:
        error = 0

        for i in range(train_data.shape[0]):
            prediction = np.dot(train_data[i], theta)

            if prediction >= 0:
                prediction = 1
            else:
                prediction = -1

            print(labels[i], prediction)

            mismatch = 1 if prediction != labels[i] else 0
            error += mismatch

            if mismatch != 0:
                print(f'{theta = }')
                theta = theta + labels[i] * train_data[i].reshape(-1, 1)
                print(f'{theta = }')

        k += error

        if error == 0:
            isConverged = True


    print(f'{isConverged = }')
    print(f'{theta = }')
    print(f'{k = }')

    return theta, k


data = np.array([[-1.939, 2.704],
                 [-0.928, -3.054],
                 [-2.181, -3.353],
                 [-0.142, 1.440],
                 [2.605, -0.651]])
labels = np.array([1, -1, -1, 1, 1])

assert data.shape[0] == labels.shape[0]

theta, k = train(data, labels)