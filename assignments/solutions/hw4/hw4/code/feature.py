import numpy as np
import sys

def split_data(file_path, delimiter='\t'):
    "Splits data into labels and raw sentence features."

    data = np.loadtxt(file_path, dtype='U', delimiter=delimiter, comments=None)

    # data has labels stored first.
    labels = data[:, 0].astype(np.int)

    # Removing unnecessary characters from the sentences.
    features = [str(line).translate({ord(i): None for i in '\"\\.,'}).strip(' ')
                for line in data[:, 1]]

    return features, labels

def get_dict_kv(dict_path, delimiter='\t'):
    "Reads the dict file and converts it to a dictionary for easy use."

    data = np.loadtxt(dict_path, dtype='U', delimiter=delimiter, comments=None)
    dictionary = {term.split(' ')[0]: int(term.split(' ')[1]) for term in data}

    return dictionary

def get_word2vec_kv(feature_dict_path, delimiter='\n'):
    "Reads the dict file and converts it to a dictionary for easy use."

    data = np.loadtxt(feature_dict_path,
                      dtype='U',
                      delimiter=delimiter,
                      comments=None)
    words = [term.split('\t')[0] for term in data]
    features = [[float(f) for f in term.split('\t')[1:]] for term in data]

    assert len(words) == len(features)

    fdictionary = {words[i]: np.array(features[i]) for i in range(len(words))}

    assert fdictionary[words[0]].shape[0] == 300

    return fdictionary

def get_bow_features(dictionary, raw_features):
    "Generates the BoW feature vector corresponding to each example."

    bow_features = np.zeros((len(raw_features), len(dictionary)), dtype=np.int)

    for i in range(len(raw_features)):

        indices = []

        for word in raw_features[i].split(' '):

            # Getting rid of empty strings.
            if not word:
                continue

            if word in dictionary:
                indices.append(dictionary[word])

        # Marking the indices correpsonding to words detected 1.
        bow_features[i][indices] = 1

    return bow_features


def get_word2vec_features(fdictionary, raw_features):
    "Generates the word2vec feature vector corresponding to each example."

    word2vec_features = np.zeros((len(raw_features),
                                  len(list(fdictionary.values())[0])))

    for i in range(len(raw_features)):

        indices = []

        for word in raw_features[i].split(' '):

            # Getting rid of empty strings.
            if not word:
                continue

            if word in fdictionary:
                # Numpy array containing the feature vector.
                indices.append(fdictionary[word])

        word2vec_features[i] = np.sum(np.stack(indices), axis=0) / len(indices)

    return word2vec_features

def write_features_to_file(file_path, features, labels, delimiter='\t'):
    """Write features and labels, in required format, to file."""

    assert features.shape[0] == labels.shape[0]

    to_write = np.hstack((labels.reshape(-1, 1), features))
    out_string = ''

    # if both labels and features are ints, we can just stack them up and
    # convert to string.
    if to_write.dtype == np.int:
        for i in range(to_write.shape[0]):
            out_string += '\t'.join(to_write[i].astype('U'))
            out_string += '\n'
    else:
        for i in range(to_write.shape[0]):
            for j in range(to_write.shape[1]):
                out_string += ('{:.6f}'.format(round(to_write[i][j], 6)) + '\t')
            out_string = out_string.strip('\t')
            out_string += '\n'

    with open(file_path, 'w') as f:
        f.write(out_string)

    print('Features written to file {0} successully!'.format(file_path))

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 9

    train_file_path = sys.argv[1]
    val_file_path = sys.argv[2]
    test_file_path = sys.argv[3]
    dict_path = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_val_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    feature_dict_path = sys.argv[9]

    print(f'{train_file_path = }')
    print(f'{val_file_path = }')
    print(f'{test_file_path = }')
    print(f'{dict_path = }')
    print(f'{formatted_train_out = }')
    print(f'{formatted_val_out = }')
    print(f'{formatted_test_out = }')
    print(f'{feature_flag = }')
    print(f'{feature_dict_path = }')

    raw_train_features, train_labels = split_data(train_file_path)
    raw_val_features, val_labels = split_data(val_file_path)
    raw_test_features, test_labels = split_data(test_file_path)

    if feature_flag == 1:
        dictionary = get_dict_kv(dict_path)

        train_features = get_bow_features(dictionary, raw_train_features)
        val_features = get_bow_features(dictionary, raw_val_features)
        test_features = get_bow_features(dictionary, raw_test_features)

    else:
        fdictionary = get_word2vec_kv(feature_dict_path)

        train_features = get_word2vec_features(fdictionary, raw_train_features)
        val_features = get_word2vec_features(fdictionary, raw_val_features)
        test_features = get_word2vec_features(fdictionary, raw_test_features)

    write_features_to_file(formatted_train_out, train_features, train_labels)
    write_features_to_file(formatted_val_out, val_features, val_labels)
    write_features_to_file(formatted_test_out, test_features, test_labels)