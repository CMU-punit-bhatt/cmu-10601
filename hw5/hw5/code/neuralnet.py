
import numpy as np
import sys

class Sigmoid:
    @classmethod
    def forward(self, x):

        return 1. / (1. + np.exp(-x))

    @classmethod
    def backward(self, a):

        return a * (1. - a)

class CrossEntropyLoss:

    @classmethod
    def forward(self, y, y_hat):

        y_hat_softmax = np.exp(y_hat) / \
            np.sum(np.exp(y_hat), axis=1).reshape(-1, 1)

        assert y.shape == y_hat_softmax.shape

        return - np.mean(np.sum(y * np.log(y_hat_softmax), axis=1))

    @classmethod
    def backward(self, y, y_hat):

        y_hat_softmax = np.exp(y_hat) / np.sum(np.exp(y_hat), axis=1).reshape(-1, 1)

        # print(f'{y_hat_softmax = }')

        return y_hat_softmax - y


class SingleLayerNeuralNet:

    def __init__(self, M, D, K, init_flag=1):

        self.M = M
        self.D = D
        self.K = K

        if init_flag == 1:
            print('\nRand Init')
            self.W1 = np.random.uniform(-0.1, 0.1, size=(D, M))
            self.W2 = np.random.uniform(-0.1, 0.1, size=(K, D))
        else:
            print('\nZero Init')

            self.W1 = np.zeros((D, M))
            self.W2 = np.zeros((K, D))

        self.b1 = np.zeros((1, D))
        self.b2 = np.zeros((1, K))
        self.s_W1 = np.zeros_like(self.W1)
        self.s_W2 = np.zeros_like(self.W2)
        self.s_b1 = np.zeros_like(self.b1)
        self.s_b2 = np.zeros_like(self.b2)

        # print(f'{self.s_W1.shape = }')
        # print(f'{self.s_W2.shape = }')
        # print(f'{self.s_b1.shape = }')
        # print(f'{self.s_b2.shape = }')

    def train(self,
              train_data,
              train_labels,
              val_data,
              val_labels,
              n_epochs,
              lr):

        train_x = train_data
        train_y = train_labels
        val_x = val_data
        val_y = val_labels

        if len(train_y.shape) == 1 or train_y.shape[1] == 1:
            train_y = np.zeros((train_y.shape[0], self.K))
            train_y[np.arange(train_labels.shape[0]), train_labels.squeeze()] = 1

        if len(val_y.shape) == 1 or val_y.shape[1] == 1:
            val_y = np.zeros((val_y.shape[0], self.K))
            val_y[np.arange(val_labels.shape[0]), val_labels.squeeze()] = 1

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        print('\nBefore Training:')
        print(f'{self.W1 = }')
        print(f'{self.b1 = }')
        print(f'{self.W2 = }')
        print(f'{self.b2 = }')

        print('\nStarting Training:\n')

        for epoch in range(n_epochs):

            for i in range(train_x.shape[0]):

                x = train_x[i].reshape(1, -1)
                y = train_y[i].reshape(1, -1)

                y_hat, loss = self.__forward(x, y)
                self.__backward(x, y, y_hat)
                self.__update_adagrad(lr)

                if i < 3:
                    print('\nAfter training {0} epoch'.format(n_epochs))
                    print(f'{self.W1 = }')
                    print(f'{self.b1 = }')
                    print(f'{self.W2 = }')
                    print(f'{self.b2 = }')

                    # print(f'{np.unique(self.W1, return_counts=True) = }')

            train_y_hat, train_loss = self.__forward(train_x, train_y)
            train_acc = self.__get_acc(train_y, train_y_hat)

            val_y_hat, val_loss = self.__forward(val_x, val_y)
            val_acc = self.__get_acc(val_y, val_y_hat)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print('{0}/{1}:'.format(epoch + 1, n_epochs))
            print('train_loss - {0}, train_acc - {1}'.format(train_loss,
                                                             train_acc))
            print('val_loss - {0}, val_acc - {1}\n'.format(val_loss, val_acc))

        return train_losses, train_accs, val_losses, val_accs

    def predict(self, data, labels):

        x = data
        y = labels

        if len(y.shape) == 1 or y.shape[1] == 1:
            y = np.zeros((y.shape[0], self.K), dtype=np.int)
            y[np.arange(y.shape[0]), labels.squeeze()] = 1
        y_hat, _ = self.__forward(x, y)

        # print(y, y_hat)
        error = 1. - self.__get_acc(y, y_hat)

        return np.argmax(y_hat, axis=1), error

    def __forward(self, x, y):

        self.z1 = x @ self.W1.T + self.b1

        self.a1 = Sigmoid.forward(self.z1)
        # z2
        y_hat = self.a1 @ self.W2.T + self.b2

        # print(f'{self.W1 = }')
        # print(f'{self.b1 = }')
        # print(f'{self.z1 = }')
        # print(f'{self.a1 = }')
        # print(f'{self.W2 = }')
        # print(f'{self.b2 = }')
        # print(f'{y_hat = }')

        # Softmax done as part of loss.

        return y_hat, self.__get_loss(y, y_hat)

    def __get_loss(self, y, y_hat):

        return CrossEntropyLoss.forward(y, y_hat)

    def __backward(self, x, y, y_hat):

        self.dz2 = CrossEntropyLoss.backward(y, y_hat)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        self.dW2 = self.dz2.T @ self.a1
        self.da1 = self.dz2 @ self.W2

        self.dz1 = Sigmoid.backward(self.a1) * self.da1
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)
        self.dW1 = self.dz1.T @ x
        self.dx = self.dz1 @ self.W1

        # print(f'{self.dz2 = }')
        # print(f'{self.db2 = }')
        # print(f'{self.dW2 = }')
        # print(f'{self.da1 = }')
        # print(f'{self.dz1 = }')
        # print(f'{self.dW1 = }')
        # print(f'{self.db1 = }')

    def __get_acc(self, y, y_hat):

        return float(np.sum(np.argmax(y, axis=1) ==
                            np.argmax(y_hat, axis=1)) / y.shape[0])

    def __update_adagrad(self, lr, eps=1e-5):

        assert self.s_W1.shape == self.dW1.shape
        assert self.s_W2.shape == self.dW2.shape
        assert self.s_b1.shape == self.db1.shape
        assert self.s_b2.shape == self.db2.shape

        self.s_W1 += (self.dW1 ** 2)
        self.s_W2 += (self.dW2 ** 2)
        self.s_b1 += (self.db1 ** 2)
        self.s_b2 += (self.db2 ** 2)

        self.W1 -= ((lr / np.sqrt(self.s_W1 + eps)) * self.dW1)
        self.W2 -= ((lr / np.sqrt(self.s_W2 + eps)) * self.dW2)
        self.b1 -= ((lr / np.sqrt(self.s_b1 + eps)) * self.db1)
        self.b2 -= ((lr / np.sqrt(self.s_b2 + eps)) * self.db2)

        # self.W1 -= (lr * self.dW1)
        # self.W2 -= (lr * self.dW2)
        # self.b1 -= (lr * self.db1)
        # self.b2 -= (lr * self.db2)

        # print('\nPost Update:')
        # print(f'{self.W1 = }')
        # print(f'{self.b1 = }')
        # print(f'{self.W2 = }')
        # print(f'{self.b2 = }')

def split_data(file_path, delimiter=','):
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

def write_metrics_to_file(file_path,
                          train_loss,
                          val_loss,
                          train_error,
                          val_error):
    """Writes train and validation metrics to file."""

    out_string = ''

    for i, (t_loss, v_loss) in enumerate(zip(train_loss, val_loss)):
        out_string += 'epoch={0} '.format(i + 1)
        out_string += 'crossentropy(train):{:.11f}\n'.format(t_loss)
        out_string += 'epoch={0} '.format(i + 1)
        out_string += 'crossentropy(validation):{:.11f}\n'.format(v_loss)

    out_string += 'error(train): {0}\n'.format(round(train_error, 3))
    out_string += 'error(validation): {0}'.format(round(val_error, 3))

    with open(file_path, 'w') as f:
        f.write(out_string)

    print('Metrics written to file {0} successully!'.format(file_path))

if __name__ == '__main__':

    assert len(sys.argv) == 1 + 9

    train_data = sys.argv[1]
    val_data = sys.argv[2]
    train_out = sys.argv[3]
    val_out = sys.argv[4]
    metrics_out = sys.argv[5]
    n_epochs = int(sys.argv[6])
    h_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    lr = float(sys.argv[9])

    print(f'{train_data = }')
    print(f'{val_data = }')
    print(f'{train_out = }')
    print(f'{val_out = }')
    print(f'{metrics_out = }')
    print(f'{n_epochs = }')
    print(f'{h_units = }')
    print(f'{init_flag = }')
    print(f'{lr = }')

    train_x, train_labels = split_data(train_data)
    val_x, val_labels = split_data(val_data)

    net = SingleLayerNeuralNet(train_x.shape[1], h_units, 4, init_flag)
    train_loss, train_acc, val_loss, val_acc = net.train(train_x,
                                                         train_labels,
                                                         val_x,
                                                         val_labels,
                                                         n_epochs,
                                                         lr)
    train_preds, train_err = net.predict(train_x, train_labels)
    val_preds, val_err = net.predict(val_x, val_labels)

    write_predictions_to_file(train_out, train_preds)
    write_predictions_to_file(val_out, val_preds)
    write_metrics_to_file(metrics_out, train_loss, val_loss, train_err, val_err)

