import numpy as np
import os
import sys

def split_data(data_file_path, delimiter='\t'):
    """Reads data from file and splits it into attribute data and labels."""

    data = np.loadtxt(data_file_path, delimiter=delimiter,  dtype='U')

    # Last column contains label. Also, ignoring first row as it contains the
    # column name.
    labels = data[1:, -1]
    data = data[1:, :-1]

    return data, labels

def train_decision_stump(train_data, labels, attribute_index):
    """Trains a decision stump based on the attribute split index passed."""

    attribute_values = np.unique(train_data[:, attribute_index])
    print(f'{attribute_values = }')

    decision_stump_dict = {}

    for value in attribute_values:
        
        # Gets the labels corresponding to an attribute value along with their
        # count.
        label_values, count = \
            np.unique(labels[train_data[:, attribute_index] == value], 
                      return_counts = True)
        
        print('Attribute_value {0} - labels {1}, count {2}'.format(value, 
                                                         label_values,
                                                         count))
        
        # Generates a decision stump attribute value to label dictionary based
        # on majority vote (label with max count).
        decision_stump_dict[value] = label_values[np.argmax(count)]
    
    print(f'{decision_stump_dict = }')

    return decision_stump_dict


def get_prediction_and_error(attribute_data, decision_stump_dict, labels):
    """Gets the predictions using the decision stump dictionary and calculates
        the error in those predictions."""
    
    assert len(attribute_data.shape) == 1
    assert len(labels.shape) == 1
    assert attribute_data.shape[0] == labels.shape[0]

    predictions = [decision_stump_dict[attribute_value] \
        for attribute_value in attribute_data]
    error = float(np.sum(predictions != labels) / labels.shape)

    return predictions, error

def write_predictions_to_file(file_path, predictions):
    """Writes prediction to file."""

    out_string = '\n'.join(predictions)

    with open(file_path, 'w') as f:
        f.write(out_string)
    
    print('Predictions written to file {0} successully!'.format(file_path))

def write_metrics_to_file(file_path, train_error, test_error):
    """Writes train and test metrics to file."""

    out_string = 'error(train): {:.6f}\n'.format(round(train_error, 6))
    out_string += 'error(test): {:.6f}'.format(round(test_error, 6))

    with open(file_path, 'w') as f:
        f.write(out_string)
    
    print('Metrics written to file {0} successully!'.format(file_path))

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 6

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out_path = sys.argv[4]
    test_out_path = sys.argv[5]
    metrics_out_path = sys.argv[6]

    print(f'{train_file_path = }')
    print(f'{test_file_path = }')
    print(f'{split_index = }')
    print(f'{train_out_path = }')
    print(f'{test_out_path = }')
    print(f'{metrics_out_path = }')

    # Getting training labels and data.

    train_data, train_labels = split_data(train_file_path)
    decision_stump_dict = train_decision_stump(train_data, 
                                               train_labels,
                                               split_index)
    train_predictions, train_error = \
        get_prediction_and_error(train_data[:, split_index],
                                 decision_stump_dict,
                                 train_labels)

    test_data, test_labels = split_data(test_file_path)
    test_predictions, test_error = \
        get_prediction_and_error(test_data[:, split_index],
                                 decision_stump_dict,
                                 test_labels)

    print(f'{train_error = }')
    print(f'{test_error = }\n')
    
    write_predictions_to_file(train_out_path, train_predictions)
    write_predictions_to_file(test_out_path, test_predictions)
    write_metrics_to_file(metrics_out_path, train_error, test_error)