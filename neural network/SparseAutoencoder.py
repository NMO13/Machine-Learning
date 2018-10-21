from my_neural_network import MyNeuralNetclass

class SparseAutoencoder(MyNeuralNetclass):
	rho = -0.9
	rho_hat = 0
	
	def __init__(self):
		pass
		
	def learn(self, epochs, eta, mini_batch_size, test_data = None):
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
				self.rho_hat = 0.999 * rho_hat + 0.001 * self.a_matrix[1]
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(i, self.evaluate(test_data), len(test_data)))
            else:
                print('Epoch {0} complete'.format(i))
				
	def update(self, w_b_gradient, eta, mini_batch_size):
        for i in self.weights[:-1]:
            tmp = self.weights[i] - eta/mini_batch_size * w_b_gradient[i]
            self.weights[i] = tmp
		
		self.weights[-1] = self.weights[-1] - eta/mini_batch_size * beta * (self.rho_hat[-1]  - self.rho[-1]
		
# The main script creates a network that learns MNIST images
if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = np.array(training_data)
    X = training_data[:, 0]
    X = np.array(list(map(lambda x: x.flatten(), X)))

    y = training_data[:, 1]
    y = np.array(list(map(lambda x: x.flatten(), y)))

    nn = SparseAutoencoder()
    nn.input = np.array(list(zip(X, y)))
    nn.add_layer(100)
    try:
        nn.learn(1500, 3, 10, test_data=test_data)
    except Exception as e:
        print(e)

	