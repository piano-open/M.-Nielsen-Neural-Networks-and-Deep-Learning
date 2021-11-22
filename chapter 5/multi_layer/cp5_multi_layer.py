
import json
import random
import time
# Third-party libraries
import numpy as np

class EarlyStop_V1(object):
    """ this is the original early stop method as described in the book """
    @staticmethod
    def es(accuracy_arr, window_len):
        if len(accuracy_arr[:-1]) >= window_len:
            previous_best = max(accuracy_arr[:-1][-window_len:])
            current_accuracy = accuracy_arr[-1]
            print("current_accuracy: {0}, stop_threshold: {1}".format(current_accuracy, previous_best))
            if current_accuracy <= previous_best:
                return True
        else:
            print("not ready for early stop")
        return False

class EarlyStop_V2(object):
    """ this is my own early stop using fix quantile """
    @staticmethod
    def es(accuracy_arr, window_len):
        quantile_lim = 0.68
        if len(accuracy_arr[:-1]) >= window_len:
            previous_best = np.quantile(accuracy_arr[:-1][-window_len:], quantile_lim)
            current_accuracy = accuracy_arr[-1]
            print("current_accuracy: {0}, stop_threshold: {1}".format(current_accuracy, previous_best))
            if current_accuracy <= previous_best:
                return True
        else:
            print("not ready for early stop")
        return False


###################################################################################################
###################################################################################################
###################################################################################################
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        expression = 0.5 * np.linalg.norm(a - y)**2
        return expression

    @staticmethod
    def delta(z, a, y):
        expression = (a-y) * sigmoid_prime(z)
        return expression

###################################################################################################
###################################################################################################
###################################################################################################
class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        expression = -y*np.log(a) - (1 - y)*np.log(1 - a)
        sum_of_ok_number = np.sum(np.nan_to_num(expression))
        return sum_of_ok_number

    @staticmethod
    def delta(z, a, y):
        return(a - y)

###################################################################################################
###################################################################################################
###################################################################################################
class Network(object):

    # modified from chapter2 code, add cross entropy related features
    def __init__(
            self,
            sizes, 
            cost=CrossEntropyCost,
            reg=None,
            early_stop_n=None,
            early_stop_method=EarlyStop_V1,
            rnd_seed=[42,42],
            mu=1.0
        ):
        self.seed1 = rnd_seed[0]
        self.seed2 = rnd_seed[1]
        print('rnd_seed[0]', rnd_seed[0])
        print('rnd_seed[1]', rnd_seed[1])

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.reg = reg
        self.early_stop_n = early_stop_n
        self.early_stop_method = early_stop_method
        self.mu = mu
        print("early_stop_method:", str(self.early_stop_method.__name__))
        print('mu=', mu)
        print('shape of network:', sizes)
        print("#=#" * 30)

    """new feature in chapter3"""
    def default_weight_initializer(self):
        np.random.seed(self.seed1)
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        """ initial weights to node n are scaled by number of connection to n """
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        # self.vel_biases = np.zeros(np.shape(self.biases))s
        # self.vel_weights = np.zeros(np.shape(self.weights))

        self.vel_biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.vel_weights = [np.zeros((y, x))
            for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    """ the simple init. weight generator in chapter 1"""
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    ###################################################################################################
    """ new SGD supporting regularization"""
    def SGD(self,
            training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):

        self.original_eta = eta

        if self.early_stop_n is not None:
            if epochs < self.early_stop_n:
                print("Error: Early stop is earlier than the first epoch")
                return

        # n = len(training_data)
        self.mini_batch_size = mini_batch_size
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for i in range(epochs):
            epoch_start_time = time.time()
            random.seed(self.seed2)
            random.shuffle(training_data)
            mini_batches = [training_data[k : k+mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)]
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
            """ original, loop format """
            # for mini_batch in mini_batches:
            #     self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
            """ matrixlization """
            input_data, target_data = [], []
            for mini_batch in mini_batches:
                input_data.append(np.column_stack([mini_batch[j][0] for j in range(mini_batch_size)]))
                target_data.append(np.column_stack([mini_batch[j][1] for j in range(mini_batch_size)]))

            for inputs, targets in zip(input_data, target_data):
                self.update_mini_batch(inputs, targets, eta, lmbda, len(training_data))

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
            epoch_end_time = time.time()
            epoch_completion_time = epoch_end_time - epoch_start_time
            print('epoch done')
            print('Epoch {0} of {1}: training complete, took {2} seconds'.format(
                i+1, epochs, round(epoch_completion_time)))

            verification_time_start = time.time()
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print('>>>> Cost on training data: {0}'.format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print('     Accuracy on training data: {0} / {1}'.format(
                    accuracy, len(training_data)))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print('>>>> Cost on evaluation data: {0}'.format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print('     Accuracy on evaluation data: {0} / {1}'.format(
                    accuracy, len(evaluation_data)))

            verification_time_end = time.time()
            verification_time = round(verification_time_end - verification_time_start)
            print("verification time:", verification_time, "s")

            self.training_cost = training_cost
            self.training_accuracy = training_accuracy
            self.evaluation_cost = evaluation_cost
            self.evaluation_accuracy = evaluation_accuracy

            peakedQ = (self.early_stop_method).es(evaluation_accuracy, self.early_stop_n)
            if peakedQ:
                print("peaked, halving eta")
                eta = eta / 2
                if self.original_eta / eta >= 128:
                    print('done')
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

            print('#'*60 + '\n')

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

###################################################################################################
    def L1_reg(self, eta, lmbda, n, w, nw):
        expr = w - np.sign(w)*(eta*lmbda/n) - (eta/self.mini_batch_size)*nw
        return expr

    def L2_reg(self, eta, lmbda, n, w, nw):
        expr = w * (1 - eta*(lmbda/n)) - (eta/self.mini_batch_size)*nw
        return expr

    def no_reg(self, eta, lmbda, n, w, nw):
        expr = w - (eta/self.mini_batch_size)*nw
        return expr

###################################################################################################
    def vel_w_L1_reg(self, eta, lmbda, n, v, nw):
        expr = (self.mu*v) - np.sign(v)*(eta*lmbda/n) - (eta/self.mini_batch_size)*nw
        return expr

    def vel_w_L2_reg(self, eta, lmbda, n, v, nw):
        expr = (self.mu*v) * (1 - eta*(lmbda/n)) - (eta/self.mini_batch_size)*nw
        return expr

    def vel_w_no_reg(self, eta, lmbda, n, v, nw):
        expr = (self.mu*v) - (eta/self.mini_batch_size)*nw
        return expr

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    """ matrixlization """
    def update_mini_batch(self, inputs, targets, eta, lmbda, n):
        nabla_b, nabla_w = self.backprop(inputs, targets)
        # print('bp done')
        # keep track of the row sum. also we want it as col vec.
        nabla_b = [np.sum(nb, axis=1).reshape((nb.shape[0],1)) for nb in nabla_b]

        if self.mu == 1.0:
            if self.reg == None:
                self.weights = [self.no_reg(eta, lmbda, n, w, nw)
                                    for w, nw in zip(self.weights, nabla_w)]
            if self.reg == "L1":
                self.weights = [self.L1_reg(eta, lmbda, n, w, nw)
                                    for w, nw in zip(self.weights, nabla_w)]
            if self.reg == "L2":
                self.weights = [self.L2_reg(eta, lmbda, n, w, nw)
                                    for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / self.mini_batch_size)*nb
                                for b, nb in zip(self.biases, nabla_b)]
        else:
            if self.reg == None:
                self.vel_weights = [self.vel_w_no_reg(eta, lmbda, n, v, nw)
                                    for v, nw in zip(self.vel_weights, nabla_w)]
            if self.reg == "L1":
                self.vel_weights = [self.vel_w_L1_reg(eta, lmbda, n, v, nw)
                                    for v, nw in zip(self.vel_weights, nabla_w)]
            if self.reg == "L2":
                self.vel_weights = [self.vel_w_L2_reg(eta, lmbda, n, v, nw)
                                    for v, nw in zip(self.vel_weights, nabla_w)]
            self.weights = np.array(self.weights) + np.array(self.vel_weights)

            self.vel_biases = [self.mu*vb - (eta/self.mini_batch_size)*nb
                                for vb, nb in zip(self.vel_biases, nabla_b)]
            self.biases = np.array(self.biases) + np.array(self.vel_biases)


#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
###################################################################################################
    """ matrizlization """
    """ new backprop """
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
        delta = (self.cost).delta(zs[-1], activations[-1], targets)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
###################################################################################################
    """ new feature in chapter 3 """
    def accuracy(self, data, convert=False):
        if convert:
            result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        else:
            result = [(np.argmax(self.feedforward(x)), y) for x, y in data]
        accuracy = sum(int(x == y) for (x, y) in result)
        return accuracy

###################################################################################################
    """ new feature in chapter 3 """
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost = cost + self.cost.fn(a, y) / len(data)
        expression = 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        cost = cost + expression
        return cost

###################################################################################################
    """ new feature in chapter 3 """
    def save(self, filename):
        data = {
                "sizes": self.sizes,
                "weights": self.weights,
                "biases": self.biases,
                "cost": str(self.cost.__name__)
                }
        with open(filename, 'w') as f:
            json.dump(data, f)

###################################################################################################
###################################################################################################
###################################################################################################

""" new feature in chapter 3 """
def load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    # get the cost function class used in data
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

###################################################################################################
def vectorized_result(n):
    e = np.zeros((10, 1))
    e[n] = 1.0
    return e

###################################################################################################
""" Miscellaneous functions """

# this has overflow problem
# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0 / (1.0 + np.exp(-z))

# Takamura san's version: limit precision to 15 decimal places
def sigmoid(x):
    input_range = 34.53877639491068407551
    x = np.clip(x, -input_range, input_range)
    return 1.0 / (1.0 + np.exp(-x))


###################################################################################################
def sigmoid_prime(z):
    """ Derivative of the sigmoid function """
    return sigmoid(z) * (1-sigmoid(z))

#############################################################################################################################
#############################################################################################################################


""" run network, chapter3 version"""
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()



early_stop_n = 10
esv = EarlyStop_V2
n_Epochs = 53
mini_batch_size = 10
eta = 0.5
r = "L1"
# r = "L2"
mu = 0.5
print('early stop length:', early_stop_n)

seed = [1234, 2021]
# seed = [42, 314]
# seed = [43, 315]

# shape = [784, 30, 10]
# shape = [784, 30, 30, 10]
shape = [784, 30, 30, 30, 30, 10]


s = time.time()
print("regularization:", r)
net = Network(
    shape,
    cost=CrossEntropyCost,
    reg=r,
    early_stop_n=early_stop_n,
    early_stop_method=esv,
    rnd_seed=seed,
    mu=mu
)
ret = net.SGD(
    training_data,
    n_Epochs,
    mini_batch_size,
    eta,
    lmbda = 1.25, # reg param.
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_accuracy=True,
    monitor_training_cost=True
    )

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = ret

# net.SGD(training_data, 180, 5, 3.0, test_data=test_data)
e = time.time()
print(e-s, 's')

print("max evaluation_accuracy:", max(evaluation_accuracy))
print("evaluation_accuracy:")
print(evaluation_accuracy)
print("=="*30)
