import random

import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_der(x):
    return 1 - (np.tanh(x)) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    # return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork(object):
    def __init__(self, layers, activation_func='tanh'):
        # 确定激活函数
        if activation_func == 'tanh':
            self.activation = tanh
            self.activation_der = tanh_der
        if activation_func == 'sigmoid':
            self.activation = sigmoid
            self.activation_der = sigmoid_der

        # 初始化两层的神经网络的权重矩阵
        self.w = list()
        self.w.append(np.random.random([layers[0] + 1, layers[1] + 1]))
        self.w.append(np.random.random([layers[1] + 1, layers[2]]))
        # print(self.w)

    def fit(self, X, Y, l=0.06, n=10000):
        """
        训练函数
        :param X: 训练矩阵
        :param Y: 标签值矩阵
        :param l: learning rate 学习率
        :param n: epochs number 迭代次数
        :return:
        """
        # 给X加一个维度，作为神经元的偏置
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        # 取所有行，少一列
        temp[:, :-1] = X
        X = temp
        # print(X)
        Y = np.array(Y)
        # print(Y)
        # 随机挑选X的一列
        for i in range(n):
            num = random.randint(0, X.shape[0] - 1)
            random_x = [X[num]]
            random_y = Y[num]
            # print(random_y)
            # print(random_x)
            # 相乘
            for dot_num in range(len(self.w)):
                random_x.append(self.activation(np.dot(random_x[dot_num], self.w[dot_num])))
                # print(random_x)

            # 求取error
            error = random_y - random_x[-1]
            # print(error)
            # 求取delta
            delta_two = error * self.activation_der(random_x[-1])
            delta_one = delta_two.dot(self.w[1].T) * self.activation_der(random_x[1])
            # print(delta_one)
            deltas = [delta_one, delta_two]
            # print(deltas)
            # 更改权重的值
            for j in range(len(self.w)):
                a = np.atleast_2d(random_x[j])
                b = np.atleast_2d(deltas[j])
                w = np.dot(a.T, b)
                self.w[j] += l * w
            # print(self.w)

    def predict(self, x):
        """预测"""
        x = np.array(x)
        print(x.shape)
        temp = np.ones([x.shape[0], x.shape[1] + 1])
        print(temp)
        temp[:, :-1] = x
        a = temp
        for l in range(0, len(self.w)):
            a = self.activation(np.dot(a, self.w[l]))
        print(a)
        return a


if __name__ == '__main__':
    print(sigmoid_der(2))
    nn = NeuralNetwork([3, 2, 1])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    nn.fit(X, [1, 0, 0])
    nn.predict(X)
