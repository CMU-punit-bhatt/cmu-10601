import numpy as np

def log_sum_exp(lx):
    """Calculates log of the sum of the exponential of input log values.

    Args:
        lx (ndarray): 1D Numpy array of log values.

    Returns:
        Log of the sum of the exponential of input log values.
    """
    assert len(lx.shape) == 1

    m = np.max(lx)

    return m + np.log(np.sum(np.exp(lx - m)))

def get_index_dictionary(filepath, delimiter='\n'):
    """Generates a dictionary of words to their respective indices.

    Args:
        filepath (str): File path.
        delimiter (str, optional): Delimiter separating words. Defaults to '\n'.

    Returns:
        Dictionary mapping words to their indices.
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    return {k.strip(delimiter): v for v, k in enumerate(lines)}

def write_matrix_to_file(filepath, matrix):
    """Write a numpy matrix to a file.

    Args:
        filepath (str): File path.
        matrix (ndarray): Numpy 2D array.
    """

    if len(matrix.shape) < 2:
        matrix = matrix.reshape(-1, 1)

    out_str = ''

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            out_str += '{:.18e} '.format(matrix[i][j])

        out_str = out_str.strip(' ')
        out_str += '\n'

    with open(filepath, 'w') as f:
        f.write(out_str)

    print('Matrix written to {0} successfully!'.format(filepath))

def read_matrix_from_file(filepath):
    """Read a numpy matrix from a file.

    Args:
        filepath (str): File path.

    Returns:
        Numpy 2D array.
    """

    matrix = np.loadtxt(filepath)

    if len(matrix.shape) < 2:
        matrix = matrix.reshape(-1, 1)

    return matrix

def write_metrics_to_file(filepath, avg_log_likelihood, accuracy):
    """Writes average log likelihood and accuracy metrics to file."""

    out_string = 'Average Log-Likelihood: {0}\n'.format(round(avg_log_likelihood,
                                                              16))
    out_string += 'Accuracy: {0}\n'.format(round(accuracy, 16))

    with open(filepath, 'w') as f:
        f.write(out_string)

    print('Metrics written to file {0} successully!'.format(filepath))

def write_predictions_to_file(filepath,
                              preds,
                              word_x,
                              y,
                              index_y,
                              split_delim='\t',
                              word_delim='\n'):
    """Writes predictions of 1 sentence to the output file.

    Args:
        filepath (str): Predictions output file path.
        preds (ndarray): Predicted label/tag indices.
        word_x (list): Actual observation/word strings.
        y (list): True label/tag indices.
        index_y (dict): Tag to index dictionary mapping
        split_delim (str, optional): Word and tag delimiter. Defaults to '\t'.
        word_delim (str, optional): Delimiter for words in a sentence.
            Defaults to '\n'.

    Returns:
        Total number of matches.
    """

    out_string = ''
    matches = int(np.sum(preds == np.array(y).flatten()))

    for i in range(len(y)):
        pred_i = list(index_y.keys())[
            list(index_y.values()).index(preds[i])]
        out_string += '{0}{1}{2}{3}'.format(word_x[i],
                                            split_delim,
                                            pred_i,
                                            word_delim)
    out_string += word_delim

    with open(filepath, 'a') as of:
        of.write(out_string)

    return matches
