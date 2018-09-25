import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            print(i)
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
        # print(self.weights)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # 一. 给X数据加一列1，相当于后续的偏置所乘的数
        X = np.atleast_2d(X)
        print(X)
        print(X.shape)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        # print(temp)
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        print(X)
        y = np.array(y)
        print(y)

        # 迭代epochs次
        for k in range(epochs):
            # 随机挑选X的一行，i为行号，a为这一行数据，为输入层数据
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # a为每层的值，a[0]为第一层输入层数据，a[1]为第二层输出层数据，a[-1]为最后一层输出层数据
            for l in range(len(self.weights)):
                # 计算每层的结果

                a.append(self.activation(np.dot(a[l], self.weights[l])))

            # Computer the error at the top layer
            # print(a)
            error = y[i] - a[-1]

            # For output layer, Err calculation (delta is updated error)
            deltas = [error * self.activation_deriv(a[-1])]

            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            # print(deltas)
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


nn = NeuralNetwork([2, 2, 1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))

# digits = load_digits()
# X = digits.data
# y = digits.target
# X -= X.min()  # normalize the values to bring them into the range 0-1
# X /= X.max()
#
# nn = NeuralNetwork([64, 100, 10], 'logistic')
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# labels_train = LabelBinarizer().fit_transform(y_train)
# labels_test = LabelBinarizer().fit_transform(y_test)
# print("start fitting")
# nn.fit(X_train, labels_train, epochs=3000)
# predictions = []
# for i in range(X_test.shape[0]):
#     o = nn.predict(X_test[i])
#     predictions.append(np.argmax(o))
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
