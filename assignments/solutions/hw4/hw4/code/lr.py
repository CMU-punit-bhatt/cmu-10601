import matplotlib.pyplot as plt
import numpy as np
import sys

class LogisiticRegression:

    def __init__(self, n_features, thres=0.5):

        self.M = n_features
        self.thres = thres

        # Including bias as theta_0.
        self.theta = np.zeros((self.M + 1, 1))

    def train_sgd(self,
                  train_features,
                  train_labels,
                  val_features,
                  val_labels,
                  n_epochs,
                  lr=0.01):

        N = train_features.shape[0]

        assert train_features.shape[1] == self.M
        assert train_labels.shape[0] == N

        assert val_features.shape[1] == self.M

        # Dont shuffle any data - autograder requirement.
        # making x_0 = 1 for all examples to accomodate for bias.
        x = np.hstack((np.ones(N).reshape(-1, 1), train_features))
        x_val = np.hstack((np.ones(val_features.shape[0]).reshape(-1, 1),
                           val_features))

        # Not making a copy as wont be modifying this.
        y = train_labels
        y_val = val_labels

        train_errs = []
        train_loss = []
        val_errs = []
        val_loss = []

        for n in range(n_epochs):
            running_loss = 0.

            for i in range(N):

                # Uncomment to calculate loss
                J_i = self.__calculate_loss(x[i], y[i])

                yhat = self.__perform_lr(x[i])
                dtheta = x[i] * (yhat - y[i]) / N
                self.theta = self.theta - lr * dtheta.reshape(-1, 1)

                running_loss += J_i

            # Predict internally assigns x_0.
            _, val_err = self.predict(val_features, val_labels)
            _, train_err = self.predict(train_features, train_labels)

            # Uncomment to capture all metrics.
            train_loss.append(self.__calculate_loss(x, y))
            val_loss.append(self.__calculate_loss(x_val, y_val))
            train_errs.append(train_err)
            val_errs.append(val_err)

            print('{0}/{1}:'.format(n + 1, n_epochs))
            print('TrainLoss - {0}, TrainErr - {1}'.format(train_loss[n],
                                                           train_errs[n]))
            print('ValLoss - {0}, ValErr - {1}\n'.format(val_loss[n],
                                                         val_errs[n]))


        return train_loss, train_errs, val_loss, val_errs

    def predict(self, x, y):

        assert len(x.shape) == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.M

        # Doesnt modify original.
        x = np.hstack((np.ones(x.shape[0]).reshape(-1, 1), x))

        preds = self.__perform_lr(x)

        preds[preds >= self.thres] = 1
        preds[preds < self.thres] = 0

        error = self.__calculate_error(preds.reshape(-1, 1), y)

        return preds.astype(np.int).reshape(-1, 1), error


    def __calculate_loss(self, x, y):

        z = np.dot(x, self.theta)
        J =  float(np.sum(np.log(1 + np.exp(z)) - (z * y)) / y.shape[0])

        return J

    def __perform_lr(self, x):
        """Performs logisitic regression and returns the sigmoid output(s)."""

        assert x.shape[-1] == self.theta.shape[0]

        return self.__sigmoid(np.dot(x, self.theta))

    def __sigmoid(self, x):
        """Calculates the elementwise sigmoid for np array x."""

        return 1. / (1. + np.exp(-x))

    def __calculate_error(self, predictions, labels):
        """Calculates prediction error."""

        return float(np.sum(predictions != labels) / labels.shape[0])

def split_data(file_path, delimiter='\t'):
    "Splits data into labels and feature vectors."

    data = np.loadtxt(file_path,
                      dtype=np.float64,
                      delimiter=delimiter,
                      comments=None)

    # data has labels stored first.
    labels = data[:, 0].astype(np.int)
    features = data[:, 1:]

    return features, labels.reshape(-1, 1)

def write_predictions_to_file(file_path, predictions):
    """Writes prediction to file."""

    out_string = '\n'.join(predictions.reshape(-1).astype('U')) + '\n'

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

    assert len(sys.argv) == 1 + 8

    formatted_train_data = sys.argv[1]
    formatted_val_data = sys.argv[2]
    formatted_test_data = sys.argv[3]
    dict_path = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    n_epochs = int(sys.argv[8])

    print(f'{formatted_train_data = }')
    print(f'{formatted_val_data = }')
    print(f'{formatted_test_data = }')
    print(f'{dict_path = }')
    print(f'{train_out = }')
    print(f'{test_out = }')
    print(f'{metrics_out = }')
    print(f'{n_epochs = }')

    train_features, train_labels = split_data(formatted_train_data)
    val_features, val_labels = split_data(formatted_val_data)
    test_features, test_labels = split_data(formatted_test_data)

    lr = LogisiticRegression(train_features.shape[1])

    train_loss, train_errs, val_loss, val_err = lr.train_sgd(train_features,
                                                             train_labels,
                                                             val_features,
                                                             val_labels,
                                                             n_epochs,
                                                             lr=0.001)

    train_preds, train_err = lr.predict(train_features, train_labels)
    test_preds, test_err = lr.predict(test_features, test_labels)

    write_predictions_to_file(train_out, train_preds)
    write_predictions_to_file(test_out, test_preds)
    write_metrics_to_file(metrics_out, train_err, test_err)

    # x = list(range(1, n_epochs + 1))
    # plt.plot(x, train_loss, 'ro-', markersize=2, label='Train')
    # plt.plot(x, val_loss, 'bo-', markersize=2, label='Validation')
    # plt.xlabel('#Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.savefig('train_v_val.png', pad_inches=0)

    # plt.show()

    np.save('{}_loss_0.001.npy'.format(formatted_train_data.split('.')[0]), np.array(train_loss))