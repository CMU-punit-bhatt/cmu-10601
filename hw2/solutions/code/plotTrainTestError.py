import decisionTree
import inspection
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 3

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    plot_out_path = sys.argv[3]

    print(f'{train_file_path = }')
    print(f'{test_file_path = }')
    print(f'{plot_out_path = }')

    train_data, train_labels, attributes = inspection.split_data(train_file_path)
    label_classes, _ = inspection.get_unique_classes_count(train_labels)
    test_data, test_labels, _ = inspection.split_data(test_file_path)

    depth_range = list(range(train_data.shape[1] + 1))
    train_errors = []
    test_errors = []

    for i in depth_range:

        decision_tree = decisionTree.DecisionTree(attributes, label_classes)
        decision_tree.train(train_data, train_labels, i)

        train_predictions, train_error = decision_tree.predict(train_data,
                                                            train_labels)
        test_predictions, test_error = decision_tree.predict(test_data,
                                                            test_labels)

        print(f'{train_error = }')
        print(f'{test_error = }\n')

        train_errors.append(train_error)
        test_errors.append(test_error)

    print(f'{depth_range = }')
    print(f'{train_errors = }')
    print(f'{test_errors = }')

    plt.plot(depth_range, train_errors, 'bo--', label = 'Train Error')
    plt.plot(depth_range, test_errors, 'ro--', label = 'Test Error')
    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Error')
    plt.title('Error on Politicians dataset')

    plt.show()
