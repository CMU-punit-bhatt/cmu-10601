import math
import numpy as np
import sys

def get_unique_classes_count(lst):
    """Loops through the list and returns the different values occuring along
        with their count as NumPy arrays."""

    unique_elements = {}

    for element in lst:
        unique_elements[element] = unique_elements.get(element, 0) + 1

    return np.array(list(unique_elements.keys())), \
        np.array(list(unique_elements.values()))

def get_entropy(labels):
    """Calculates entropy using the formula
        `-Sum(Prob(class) * log2(Prob(class)))` for each class in labels."""

    assert len(labels.shape) == 1

    _, count = get_unique_classes_count(labels)
    probabilities = count / labels.shape

    return -np.sum(probabilities * np.log2(probabilities))

def get_error_rate(labels):
    """Calculates error rate using majority voting."""

    assert len(labels.shape) == 1

    _, count = get_unique_classes_count(labels)

    return float((labels.shape - np.max(count)) / labels.shape)

def split_data(data_file_path, delimiter='\t'):
    """Reads data from file and splits it into attribute data and labels."""

    data = np.loadtxt(data_file_path, delimiter=delimiter,  dtype='U')

    # Last column contains label. Also, ignoring first row as it contains the
    # column name.
    labels = data[1:, -1]
    column_heads = data[0, :-1]
    data = data[1:, :-1]

    return data, labels, column_heads

def write_metrics_to_file(file_path, entropy, error):
    """Writes error and entropy metrics to file."""

    print(f'{entropy = }')
    print(f'{error = }')

    out_string = 'entropy: {:.12f}\n'.format(round(entropy, 12))
    out_string += 'error: {:.12f}'.format(round(error, 12))

    with open(file_path, 'w') as f:
        f.write(out_string)

    print('Metrics written to file {0} successully!'.format(file_path))

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 2

    train_file_path = sys.argv[1]
    inspect_out_path = sys.argv[2]

    print(f'{train_file_path = }')
    print(f'{inspect_out_path = }')

    # Getting training labels and data.

    train_data, train_labels, _ = split_data(train_file_path)
    write_metrics_to_file(inspect_out_path,
                          get_entropy(train_labels),
                          get_error_rate(train_labels))
