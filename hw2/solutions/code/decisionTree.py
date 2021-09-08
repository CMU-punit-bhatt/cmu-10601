import sys
import numpy as np

from inspection import get_entropy, split_data, get_unique_classes_count

class DecisionTree:

    def __init__(self, attribute_names, label_classes):
        """Initializes a new instance of Decision Tree

        Args:
            attribute_names: List of attribute names
            label_classes: List of label class values
        """

        self.__attribute_names = attribute_names
        self.__label_classes = sorted(label_classes)
        self.__root_node = None

    def train(self, train_data, labels, max_depth):
        """Trains the decision tree up to the maximum allowed depth."""

        attributes_used = [False for i in range(len(self.__attribute_names))]

        self.__root_node = self.__train_tree(train_data,
                                             labels,
                                             max_depth,
                                             0,
                                             attributes_used)

        print('\nTraining completed!\n')

    def predict(self, data, labels = None):
        """Predicts the label for the data using the decision tree and
            calculates the error as well."""

        assert len(data.shape) == 2
        assert labels is None or len(labels.shape) == 1
        assert self.__root_node is not None

        predictions = np.array([self.__predict_single(data[i]) \
            for i in range(data.shape[0])])

        error = None

        if labels is not None:
            error = self.__calculate_error(predictions, labels)

        return predictions, error

    def print_tree(self, console_print = True):
        """Prints the tree in the required format."""

        out_str = self.__print_helper(self.__root_node) + '\n'

        if console_print:
            print(out_str)

        return out_str

    def __print_helper(self,
                       node,
                       split_attribute_name = None,
                       split_attribute_class = None):
        """Helper method to print tree in required format."""

        out_str = ''

        count_str = '['
        label_counts_dict = node.get_label_counts()

        for label_class in self.__label_classes:
            count = 0

            if label_counts_dict.get(label_class) is not None:
                count = label_counts_dict.get(label_class)

            count_str += str(count) + ' ' + str(label_class) + '/'

        count_str = count_str.rstrip('/')
        count_str += ']'

        depth = node.get_depth()

        if depth > 0:
            out_str += ('| ' * depth)
            out_str += split_attribute_name + ' = '
            out_str += split_attribute_class + ': '

        out_str += count_str + '\n'

        branches = node.get_branches()

        if branches is None:
            return out_str

        for attribute_class in sorted(branches.keys()):

            next_node = branches[attribute_class]
            out_str += self.__print_helper(next_node,
                                           self.__attribute_names[
                                               node.get_split_attribute()],
                                           attribute_class)

        return out_str

    def __predict_single(self, row_data):
        """Predicts single data row by traversing through the tree."""

        assert len(row_data.shape) == 1

        node = self.__root_node
        split_attribute = node.get_split_attribute()

        # Split attribute would be None only in case of leaf node.
        while(split_attribute is not None):

            attribute_class = row_data[split_attribute]

            # Getting the next node based on the value of the attribute in the
            # data row.
            node = node.get_branch(attribute_class)
            split_attribute = node.get_split_attribute()

        return node.get_best_label()

    def __calculate_error(self, predictions, labels):
        """Calculates prediction error."""

        return float(np.sum(predictions != labels) / labels.shape)

    def __get_mutual_information(self, attribute_data, labels):
        """Calculates mutual information using the formula =>
            H(Y; D) - Sum(Prob(attribute_class) * H(Y; D(attr = attr_class))),
            for each attribute class."""

        assert len(attribute_data.shape) == 1
        assert len(labels.shape) == 1

        mutual_information = get_entropy(labels)

        attribute_classes, classes_count = np.unique(attribute_data,
                                                    return_counts = True)
        total_count = np.sum(classes_count)

        for attr_class, count in zip(attribute_classes, classes_count):
            class_probability = count / total_count

            mutual_information -= class_probability * \
                get_entropy(labels[attribute_data == attr_class])

        return mutual_information

    def __get_best_attribute(self, train_data, labels, attributes_used):
        """Figures out the best attribute for splitting by calculating mutual
            information for each attribute."""

        assert len(train_data.shape) == 2

        max_mutual_information = None
        selected_attribute = None

        for i in range(train_data.shape[1]):

            # Skipping if attribute already used.
            if attributes_used[i] == True:
                continue

            mutual_information = self.__get_mutual_information(train_data[:, i],
                                                               labels)

            if max_mutual_information is None or \
                mutual_information > max_mutual_information:

                max_mutual_information = mutual_information
                selected_attribute = i

        return selected_attribute, max_mutual_information

    def __train_tree(self,
                     train_data,
                     labels,
                     max_depth,
                     curr_depth,
                     attributes_used):
        """Recursively trains a tree using further filtered data and labels at
            each step/level.

        Args:
            train_data: Filtered training data containing all
                columns/attributes but selected rows based on the parent node
                attribute and its class - whether 'y' or 'n'.
            labels: Filtered labels
            max_depth: Maximum allowed depth
            curr_depth: Current depth.
            attributes_used: Attributes used already in the current tree path.

        Returns:
            Sub-tree root node.
        """

        curr_node = DecisionNode(labels, curr_depth)

        # Returning if max depth reached or all attributes used up in the path.
        if max_depth == curr_depth or False not in attributes_used:
            return curr_node

        best_attribute, mutual_information = \
            self.__get_best_attribute(train_data, labels, attributes_used)

        # Returning if mutual info has been reduced to zero or there is no
        # longer any info gain with splitting.
        if mutual_information <= 0:
            return curr_node

        # Marking attribute as used and selecting it as the node splitting
        # attribute.
        attributes_used[best_attribute] = True
        curr_node.set_split_attribute(best_attribute)

        attribute_classes, classes_count = \
            get_unique_classes_count(train_data[:, best_attribute])

        for attribute_class, count in zip(attribute_classes, classes_count):

            # A row mask based on the attribute class chosen.
            row_mask = train_data[:, best_attribute] == attribute_class

            curr_node.add_node(attribute_class,
                               self.__train_tree(train_data[row_mask],
                                                 labels[row_mask],
                                                 max_depth,
                                                 curr_depth + 1,
                                                 attributes_used))

        # Freeing up the attribute used here so that any other branch not in
        # this path can use it.
        attributes_used[best_attribute] = False

        return curr_node

class DecisionNode:

    def __init__(self, labels, depth):
        """Initializes a new instance of Decision Node."""

        self.__split_attribute = None
        self.__label_counts = {}
        self.__best_label = None
        self.__branches = None
        self.__depth = depth

        self.__set_node_labels(labels)

    def add_node(self, attribute_class, node):
        """Adds a node based on the splitting attribute's class - eg. 'y'/'n'.
        """

        if self.__branches is None:
            self.__branches = {}

        self.__branches[attribute_class] = node

    def set_split_attribute(self, attribute_index):
        """Sets the splitting attribute index"""

        self.__split_attribute = attribute_index

    def get_best_label(self):
        """Gets the best label calculated"""

        return self.__best_label

    def get_split_attribute(self):
        """Gets the attribute index based on which this current node will
            further split the data. Will be None for leaf nodes."""

        return self.__split_attribute

    def get_depth(self):
        """Gets the depth of the current node."""

        return self.__depth

    def get_label_counts(self):
        """Gets the count of different labels at the current node -
            eg. [5 democrat/ 1 republican]."""

        return self.__label_counts

    def get_branches(self):
        """Gets the branches splitting from the current node."""

        return self.__branches

    def get_branch(self, attribute_class):
        """Gets current node's splitting branch for the attribute class passed
            eg. 'y'/'n'."""

        assert self.__branches is not None
        assert self.has_branch(attribute_class)

        return self.__branches[attribute_class]

    def has_branch(self, attribute_class):
        """Checks if current node has a branch for the attribute class
            eg. 'y'/'n'."""

        return self.__branches is not None and \
            self.__branches.get(attribute_class) is not None

    def __set_node_labels(self, labels):
        """
        Takes the list of labels and populates the internal label_counts
        dictionary with mapping between label classes/values and their
        count for current node.
        Also, calculates the best label based on majority voting (highest count).
        In case of labels with same votes/counts, the one with lexicographically
        last name would be chosen.

        """

        assert len(labels.shape) == 1

        label_classes, label_counts = get_unique_classes_count(labels)

        for label_class, label_count in zip(label_classes, label_counts):

            self.__label_counts[label_class] = label_count

            if self.__best_label is None or \
                self.__label_counts[self.__best_label] < label_count:

                self.__best_label = label_class
                continue


            # In case of labels with same count/vote, go with the one
            # which is lexicographically last.
            if self.__label_counts[self.__best_label] == label_count:
                self.__best_label = \
                    sorted([self.__best_label, label_class])[-1]

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


def write_tree_to_file(file_path, decision_tree):
    """Writes decision tree to file."""

    out_string = decision_tree.print_tree(False).rstrip('\n')

    with open(file_path, 'w') as f:
        f.write(out_string)

    print('Decision Tree written to file {0} successully!'.format(file_path))

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 6

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out_path = sys.argv[4]
    test_out_path = sys.argv[5]
    metrics_out_path = sys.argv[6]

    print(f'{train_file_path = }')
    print(f'{test_file_path = }')
    print(f'{max_depth = }')
    print(f'{train_out_path = }')
    print(f'{test_out_path = }')
    print(f'{metrics_out_path = }')

    train_data, train_labels, attributes = split_data(train_file_path)
    label_classes, _ = get_unique_classes_count(train_labels)
    test_data, test_labels, _ = split_data(test_file_path)

    decision_tree = DecisionTree(attributes, label_classes)
    decision_tree.train(train_data, train_labels, max_depth)

    train_predictions, train_error = decision_tree.predict(train_data,
                                                           train_labels)
    test_predictions, test_error = decision_tree.predict(test_data,
                                                         test_labels)

    print(f'{train_error = }')
    print(f'{test_error = }\n')

    write_predictions_to_file(train_out_path, train_predictions)
    write_predictions_to_file(test_out_path, test_predictions)
    write_metrics_to_file(metrics_out_path, train_error, test_error)
    write_tree_to_file('tree.txt', decision_tree)
