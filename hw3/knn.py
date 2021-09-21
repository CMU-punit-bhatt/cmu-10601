import numpy as np

def predict(test_pts, test_labels, train_pts, train_labels, k):

    predictions = []

    for i in range(test_pts.shape[0]):

        pt = test_pts[i]
        dists = np.sqrt(np.sum((train_pts - pt) ** 2, axis = 1))
        print(f'\n{dists = }')
        print(f'{labels = }')
        sorted_labels = (train_labels[np.argsort(dists)])[:k]

        preds, counts = np.unique(sorted_labels, return_counts=True)

        predictions.append(preds[np.argmax(counts)])

    predictions = np.array(predictions)
    # print(predictions != test_labels)
    error = float(np.sum(predictions != test_labels)) / test_labels.shape[0]

    return predictions, error

data = np.array([[5, 1],
                 [6, 2],
                 [7, 3],
                 [8, 4],
                 [9, 5],
                 [7, 2],
                 [8, 3]])
                #  [1, 5],
                #  [2, 6],
                #  [3, 7],
                #  [4, 8],
                #  [5, 9],
                #  [2, 7],
                #  [3, 8]])
labels = np.array([1, 1, 1, 1, 1, 0, 0])
                #    0, 0, 0, 0, 0, 1, 1])

test = np.array([[3, 9]])
test_labels = np.array([0])

k = 1
preds, error = predict(test, test_labels, data, labels, k=k)

print(f'{k = }')
print(f'{error = }')
print(f'{preds = }')

k = 5
preds, error = predict(test, test_labels, data, labels, k=k)

print(f'{k = }')
print(f'{error = }')
print(f'{preds = }')

k = 9
preds, error = predict(test, test_labels, data, labels, k=k)

print(f'{k = }')
print(f'{error = }')
print(f'{preds = }')

k = 12
preds, error = predict(test, test_labels, data, labels, k=k)

print(f'{k = }')
print(f'{error = }')
print(f'{preds = }')

# k = 2
# preds, error = predict(data, labels, data, labels, k=k)

# print(f'{k = }')
# print(f'{error = }')
# print(f'{preds = }')

# k = 1
# preds, error = predict(data, labels, data, labels, k=k)

# print(f'{k = }')
# print(f'{error = }')
# print(f'{preds = }')
