import numpy as np
import sys

from utils import *

def predict_one(x, K, A, B, C):
    """Predicts the output tag for each of the x observations by precomputing
    alpha and beta matrices.

    Args:
        x (list): Observations
        K (int): Number of states
        A (ndarray): Emission matrix
        B (ndarray): Transition matrix
        C (ndarray): Initial state probability matrix

    Returns:
        Predictions and log likelihood.
    """

    T = len(x)
    lalpha = np.zeros((K, T))
    lbeta = np.zeros((K, T))
    # alpha = np.zeros((K, T))
    # beta = np.zeros((K, T))

    lalpha[:, 0] = np.log(C[:, 0]) + np.log(A[:, x[0]])
    lbeta[:, -1] = np.log(1)

    # alpha [:, 0] = C[:, 0] * A[:, x[0]]
    # beta[:, -1] = 1

    # Pre computing all alphas in log space.
    for t in range(1, T):
        for k in range(K):
            lalpha[k, t] = np.log(A[k, x[t]]) + \
                log_sum_exp(np.log(B[:, k]) + lalpha[:, t - 1])
            # alpha[k, t] = A[k, x[t]] * \
            #     np.sum(B[:, k] * alpha[:, t - 1])

    # Pre computing all betas in log space.
    for t in reversed(range(T - 1)):
        for k in range(K):
            lbeta[k, t] = \
                log_sum_exp(lbeta[:, t + 1] + np.log(A[:, x[t + 1]]) + \
                    np.log(B[k, :]))
            # beta[k, t] = \
            #     np.sum(beta[:, t + 1] * A[:, x[t + 1]] * B[k, :])

    probs = lalpha + lbeta
    preds = np.argmax(probs, axis = 0)
    log_likelihood = float(log_sum_exp(lalpha[:, -1]))

    return preds, log_likelihood

def predict_all(val_path,
                predictions_out,
                index_y,
                index_x,
                A,
                B,
                C,
                split_delim='\t',
                word_delim='\n'):
    """Predicts the output for every sentence and writes the predictions to an
    output file.

    Args:
        val_path (str): Validation data file path
        predictions_out (str): Predictions output file path
        index_y (dict): Tag to index dictionary mapping
        index_x (dict): Word to index dictionary mapping
        A (ndarray): Emission matrix
        B (ndarray): Transition matrix
        C (ndarray): Initial state probability matrix
        split_delim (str, optional): Word and tag delimiter. Defaults to '\t'.
        word_delim (str, optional): Delimiter for words in a sentence.
            Defaults to '\n'.

    Raises:
        RuntimeError: Index_Y key not found
        RuntimeError: Index_X key not found

    Returns:
        Average log likelihood per sequence/sentence and accuracy of word tag
        matches.
    """

    with open(val_path, 'r') as f:
        line = f.readline()
        x = []
        y = []
        word_x = []
        total_words = 0
        total_sentences = 0
        total_matches = 0
        total_log_likelihood = 0.

        # Clearing predictions output file.
        with open(predictions_out, 'w') as of:
            of.write('')

        while line:

            if line == word_delim:

                preds, log_likelihood = predict_one(x, len(index_y), A, B, C)

                line = f.readline()
                matches = write_predictions_to_file(predictions_out,
                                                    preds,
                                                    word_x,
                                                    y,
                                                    index_y,
                                                    split_delim,
                                                    word_delim)

                x = []
                y = []
                word_x = []

                total_matches += matches
                total_log_likelihood += log_likelihood
                total_sentences += 1

                continue

            line = line.strip(word_delim)
            split = line.split(split_delim)

            split = [word.strip(' ') for word in split]

            assert len(split) == 2

            if split[0] not in index_x:
                raise RuntimeError('"{0}" not in INDEX_X'.format(split[0]))
            if split[1] not in index_y:
                raise RuntimeError('"{0}" not in INDEX_Y'.format(split[1]))

            x.append(index_x[split[0]])
            y.append(index_y[split[1]])
            word_x.append(split[0])

            line = f.readline()
            total_words += 1

        preds, log_likelihood = predict_one(x, len(index_y), A, B, C)
        matches = write_predictions_to_file(predictions_out,
                                            preds,
                                            word_x,
                                            y,
                                            index_y,
                                            split_delim,
                                            word_delim)
        total_matches += matches
        total_log_likelihood += log_likelihood
        total_sentences += 1

    avg_log_likelihood = total_log_likelihood / total_sentences
    accuracy = total_matches / total_words

    return avg_log_likelihood, accuracy

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 8

    val_path = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit_out = sys.argv[4]
    hmmemit_out = sys.argv[5]
    hmmtrans_out = sys.argv[6]
    predictions_out = sys.argv[7]
    metrics_out = sys.argv[8]

    print(f'{val_path = }')
    print(f'{index_to_word = }')
    print(f'{index_to_tag = }')
    print(f'{hmminit_out = }')
    print(f'{hmmemit_out = }')
    print(f'{hmmtrans_out = }')

    index_y = get_index_dictionary(index_to_tag)
    index_x = get_index_dictionary(index_to_word)

    A = read_matrix_from_file(hmmemit_out)
    B = read_matrix_from_file(hmmtrans_out)
    C = read_matrix_from_file(hmminit_out)

    avg_log_likelihood, accuracy = predict_all(val_path,
                                               predictions_out,
                                               index_y,
                                               index_x,
                                               A,
                                               B,
                                               C)
    write_metrics_to_file(metrics_out, avg_log_likelihood, accuracy)
