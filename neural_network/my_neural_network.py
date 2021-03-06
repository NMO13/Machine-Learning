# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Configurable neural network that uses vanilla batch gradient descent
class MyNeuralNet:
    def __init__(self):
        self.weights = list()
        self.last_layer_neuron_count = 0

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value
        self.last_layer_neuron_count = value[0][0].shape[0]

    def add_layer(self, neuron_count, seed = None):
        np.random.seed(seed)
        weight_layer = np.random.randn(neuron_count, self.last_layer_neuron_count + 1) / neuron_count
        weight_layer[:, -1] = 0

        # add a new layer of neurons and initialize weights randomly
        self.weights.append(weight_layer)
        self.last_layer_neuron_count = neuron_count

    def learn(self, epochs, eta, mini_batch_size, test_data = None, activation='sigmoid'):
        self.activation = activation
        if self.last_layer_neuron_count == 0:
            raise ValueError('No output layer specified.')
        for i in range(epochs):
            np.random.shuffle(self.input)
            mini_batches = [self._input[k:k+mini_batch_size] for k in range(0, len(self._input), mini_batch_size)]
            for mini_batch in mini_batches:
            # apply gradient descent on cost function and
            # update all weights
                self.feed_forward(mini_batch[:,0])
                res = self.backward(mini_batch[:,1])
                self.update(res, eta, mini_batch_size)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i, self.evaluate(test_data), len(test_data)))
            else:
                print('Epoch {0} complete'.format(i))

    def classify(self, input):
        if len(self.weights) == 0:
            raise ValueError('Network is not initialized.')
        a = input
        for w in self.weights:
            a = np.append(a, [[1] for x in a], axis=1)
            z = np.dot(a, w.T)
            a = self.sigmoid(z) if self.activation == 'sigmoid' else self.relu(z)
        return a

    def feed_forward(self, mini_batch):
        mini_batch = np.stack(mini_batch, axis=0)
        a = np.array(mini_batch, copy=True)
        self.a_matrix = [a]
        self.z_matrix = [[]] # input layer does not calculate a linear output
        for w in self.weights:
            # add bias term
            a = np.append(a, [[1] for x in a], axis=1)
            z = np.dot(a, w.T)
            self.z_matrix.append(z)
            a = self.sigmoid(z) if self.activation == 'sigmoid' else self.relu(z)
            self.a_matrix.append(a)
        self.output = a
        #todo
        #print(self.calc_loss())

    def backward(self, y):
        y = np.stack(y, axis=0)
        w_b_gradient = [np.zeros(w.shape) for w in self.weights]
        if self.a_matrix[-1].shape != y.shape:
            raise ValueError('The dimensions of the y and output layer do not match')
        ############# output layer
        # 1. how much did we miss in the output neuron?
        error = self.a_matrix[-1] - y

        # 2. how much does sigma change?
        z_prime = self.a_matrix[-1] * (1 - self.a_matrix[-1]) if self.activation == 'sigmoid' else self.relu_deriv(self.z_matrix[-1])

        # 3. calculate delta term for output layer L
        delta_L = (error * z_prime)

        # 4. multiply each output with delta term
        # and save new weights in matrix
        w_b_gradient[-1][:, :-1] = np.dot(delta_L.T, self.a_matrix[-2])

        # save new biases as well
        w_b_gradient[-1][:, -1] = np.sum(delta_L.T, axis=1)


        last_delta = delta_L
        ########### hidden layers
        for n in range(len(self.weights) - 2, -1, -1):
            weights_prev_layer = self.weights[n + 1][:, :-1]
            z_prime = self.a_matrix[n + 1] * (1 - self.a_matrix[n + 1]) if self.activation == 'sigmoid' else self.relu_deriv(self.z_matrix[n + 1])
            delta_l2 = np.dot(last_delta, weights_prev_layer) * z_prime
            w_b_gradient[n][:, :-1] = np.dot(delta_l2.T, self.a_matrix[n])
            w_b_gradient[n][:, -1] = np.sum(delta_l2.T, axis=1)
            last_delta = delta_l2

        return w_b_gradient

    def sigmoid(self, z):
        # see https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
        #z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return z * (z > 0)

    def relu_deriv(self, x):
        return 1. * (x > 0)

    def update(self, w_b_gradient, eta, mini_batch_size):
        for i in range(len(self.weights)):
            tmp = self.weights[i] - eta/mini_batch_size * w_b_gradient[i]
            self.weights[i] = tmp

    # method from http://neuralnetworksanddeeplearning.com/chap1.html
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.classify(np.array([x.flatten()]))), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# The main script creates a network that learns MNIST images
if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = np.array(training_data)
    X = training_data[:, 0]
    X = np.array(list(map(lambda x: x.flatten(), X)))

    y = training_data[:, 1]
    y = np.array(list(map(lambda x: x.flatten(), y)))

    nn = MyNeuralNet()
    nn.input = np.array(list(zip(X, y)))
    nn.add_layer(100)
    nn.add_layer(10)
    try:
        nn.learn(1500, 0.01, 30, test_data=test_data, activation='relu')
    except Exception as e:
        print(e)
