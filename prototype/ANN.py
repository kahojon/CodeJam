import json
import random
import sys
import numpy as np

class CrossEntropyCost(object):
    def cross_entropy(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def cost_der(z, a, y):
        return (a - y[:, None])

class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feed(self, a):
        a = a[None, :].transpose()
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, mini_batch_s, eta):
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_s]
                for k in xrange(0, n, mini_batch_s)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta, len(training_data))
            print "Epoch %s of %s complete" % (j, str(epochs))

    def update(self, mini_batch, eta, n):
        nbs = [np.zeros(b.shape) for b in self.biases]
        nws = [np.zeros(w.shape) for w in self.weights]
        for y, x in mini_batch:
            delta_nbs, delta_nws = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nbs, delta_nbs)]
            nabla_w = [nw + dnw for nw, dnw in zip(nws, delta_nws)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nws)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nbs)]

    def backprop(self, x, y):
        x = x[None, :].transpose()
        nbs = [np.zeros(b.shape) for b in self.biases]
        nws = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (self.cost).cost_der(zs[-1], activations[-1], y)
        nbs[-1] = delta
        nws[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nbs[-l] = delta
            nws[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nbs, nws)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# def save_instance(filename, n):
#    with open(filename,'rw') as f:
#        f.write(np.array_str)

def main():
    from read import getData as gd
    g = gd()

    tset = g.get_trainset(0.8, 265, vectorize=True)
    train_data = tset[0]
    test_data = tset[1]

    n = Network([265, 100, 100, 2])
    # train
    n.train(train_data, 2000, 20, 0.0001)
    for i in xrange(len(test_data)):
        print n.feed(test_data[i][1])
        print test_data[i][0]
        # n.save

if __name__ == '__main__':
    main()
