import numpy as np
import sys

from utils import *

def learn_hmm(train_path, index_y, index_x, split_delim='\t', word_delim='\n'):
    """Reads the training data and learns the emission, transition and initial
    probability matrices.

    Args:
        train_path (str): Train data file path
        index_y (dict): Tag to index dictionary mapping
        index_x (dict): Word to index dictionary mapping
        split_delim (str, optional): Word and tag delimiter. Defaults to '\t'.
        word_delim (str, optional): Delimiter for words in a sentence.
            Defaults to '\n'.

    Raises:
        RuntimeError: Index_Y key not found
        RuntimeError: Index_X key not found

    Returns:
        The emission (A), transition (B) and initial probability (C) matrices.
    """

    # To avoid divide by 0.
    A = np.ones((len(index_y), len(index_x)))
    B = np.ones((len(index_y), len(index_y)))
    C = np.ones((len(index_y), 1))

    with open(train_path, 'r') as f:
        line = f.readline()
        isNewSentence = True
        prev_y = None

        while line:

            if line == word_delim:
                line = f.readline()
                isNewSentence = True
                prev_y = None
                continue

            line = line.strip(word_delim)
            split = line.split(split_delim)

            split = [word.strip(' ') for word in split]

            assert len(split) == 2

            if split[0] not in index_x:
                raise RuntimeError('"{0}" not in INDEX_X'.format(split[0]))
            if split[1] not in index_y:
                raise RuntimeError('"{0}" not in INDEX_Y'.format(split[1]))

            x = index_x[split[0]]
            y = index_y[split[1]]

            if isNewSentence:
                C[y] += 1
            else:
                B[prev_y, y] += 1

            A[y, x] += 1

            line = f.readline()
            isNewSentence = False
            prev_y = y

    ## Converting to probabilities.
    # P(X_t | Y_t) = N(X_t, Y_t) / N(Y_t)
    # P(Y_t | Y_t-1) = N(Y_t, Y_t-1) / N(Y_t-1)
    # P(Y_1) = N(Y_1) / N_1 or number of lines.
    A = A / np.sum(A, axis = 1, keepdims=True)
    B = B / np.sum(B, axis = 1, keepdims=True)
    C = C / np.sum(C)

    return A, B, C

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 6

    train_path = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit_out = sys.argv[4]
    hmmemit_out = sys.argv[5]
    hmmtrans_out = sys.argv[6]

    print(f'{train_path = }')
    print(f'{index_to_word = }')
    print(f'{index_to_tag = }')
    print(f'{hmminit_out = }')
    print(f'{hmmemit_out = }')
    print(f'{hmmtrans_out = }')

    index_y = get_index_dictionary(index_to_tag)
    index_x = get_index_dictionary(index_to_word)

    A, B, C = learn_hmm(train_path, index_y, index_x)

    write_matrix_to_file(hmmemit_out, A)
    write_matrix_to_file(hmmtrans_out, B)
    write_matrix_to_file(hmminit_out, C)