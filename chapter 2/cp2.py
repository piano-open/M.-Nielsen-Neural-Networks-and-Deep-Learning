
# Problem:
# Fully matrix-based approach to backpropagation over a mini-batch Our implementation of stochastic gradient descent loops over training examples in a mini-batch.

import numpy as np
import random

random.seed(1234)
np.random.seed(1234)

class Network():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epoches, mini_batch_size, eta, test_data=None):
        self.mini_batch_size = mini_batch_size
        training_data = list(training_data)
        n = len(list(training_data))
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for i in range(epoches):
            random.shuffle(training_data)
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            input_data, target_data = [], []
            for mini_batch in mini_batches:
                # expand input data in row direction (add to matrix as new cols)
                input_data.append(np.column_stack([mini_batch[j][0] for j in range(mini_batch_size)]))
                # expand target data in row direction (add to matrix as new cols)
                target_data.append(np.column_stack([mini_batch[j][1] for j in range(mini_batch_size)]))
            for inputs, targets in zip(input_data, target_data):
                    self.update_mini_batch(inputs, targets, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(i))

    def update_mini_batch(self, inputs, targets, eta):
        nabla_b, nabla_w = self.backprop(inputs, targets)
        # keep track of the row sum. also we want it as col vec.
        nabla_b = [np.sum(nb, axis=1).reshape((nb.shape[0],1)) for nb in nabla_b]
        self.weights = [w - (eta / self.mini_batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / self.mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, inputs, targets):
        # expand b in row direction (add as new cols)
        nabla_b = [np.tile(np.zeros(b.shape), (1, self.mini_batch_size)) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = inputs
        activations = [inputs]
        zs = []
        for b, w in zip(self.biases, self.weights):
            bias = np.tile(b, (1, self.mini_batch_size))
            z = np.dot(w, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], targets) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, targets):
        return (output_activations - targets)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

##############################################################################
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

##############################################################################


""" run network """
import time
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

s = time.time()
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
e = time.time()


print(e-s, 's')
end = input("done")






