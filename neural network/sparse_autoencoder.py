from my_neural_network import MyNeuralNet
import numpy as np

class SparseAutoencoder(MyNeuralNet):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

    def learn(self, epochs, eta, mini_batch_size, test_data = None):
        if len(self.weights) != 2:
            raise ValueError('Exactly 3 layers are required.')
        if self.last_layer_neuron_count == 0:
            raise ValueError('No output layer specified.')

        hidden_layer_neurorn_count = self.weights[0].shape[0]
        self.rho_hat = np.zeros(hidden_layer_neurorn_count)
        self.rho = np.repeat(-0.9, hidden_layer_neurorn_count)
        self.beta = np.ones(hidden_layer_neurorn_count) * 0.1
        for i in range(epochs):
            np.random.shuffle(self.input)
            mini_batches = [self._input[k:k+mini_batch_size] for k in range(0, len(self._input), mini_batch_size)]
            for mini_batch in mini_batches:
            # apply gradient descent on cost function and
            # update all weights
                self.feed_forward(mini_batch[:,0])
                res = self.backward(mini_batch[:,1])

                for a in self.a_matrix[1]:
                    self.rho_hat = 0.999 * self.rho_hat + 0.001 * a
                self.update(res, eta, mini_batch_size)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i, self.evaluate(test_data), len(test_data)))
            else:
                print('Epoch {0} complete'.format(i))

    def update(self, w_b_gradient, eta, mini_batch_size):
        self.weights[0][:, :-1] = self.weights[0][:, :-1] - eta / mini_batch_size * w_b_gradient[0][:, :-1]
        self.weights[0][:, -1] = self.weights[0][:, -1] - eta / mini_batch_size * self.beta * (self.rho_hat - self.rho)

        self.weights[1] = self.weights[1] - eta/mini_batch_size * w_b_gradient[1]


# The main script creates a network that learns MNIST images
if __name__ == '__main__':
    import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = np.array(training_data)
    X = training_data[:, 0]
    X = np.array(list(map(lambda x: x.flatten(), X)))

    y = training_data[:, 1]
    y = np.array(list(map(lambda x: x.flatten(), y)))

    X = np.array([[2, 3], [4, 5], [7, 8]])
    nn = SparseAutoencoder()
    nn.input = np.array(list(zip(X, X)))
    nn.add_layer(5)
    nn.add_layer(2)
    try:
        nn.learn(1500, 3, 10)
        nn.classify([[2, 3]])
    except Exception as e:
        print(e)

