from my_neural_network import MyNeuralNet
import numpy as np
import matplotlib.pyplot as plt

class SparseAutoencoder(MyNeuralNet):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

    def learn(self, epochs, eta, mini_batch_size, test_data = None, activation='sigmoid'):
        if len(self.weights) != 2:
            raise ValueError('Exactly 3 layers are required.')
        if self.last_layer_neuron_count == 0:
            raise ValueError('No output layer specified.')
        self.activation = activation

        hidden_layer_neurorn_count = self.weights[0].shape[0]
        self.rho_hat = np.zeros(hidden_layer_neurorn_count)
        self.rho = np.repeat(0.01, hidden_layer_neurorn_count)
        self.beta = np.ones(hidden_layer_neurorn_count) * 3
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

def show_image(input_image, res_image):
    fig = plt.figure(figsize=(10, 10))
    pixels = input_image[0].flatten()
    label = input_image[1]
    pixels = pixels.reshape((28, 28))
    # Plot
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(label)
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')

    pixels = res_image[0].flatten()
    pixels = pixels.reshape((28, 28))
    # Plot
    fig.add_subplot(1, 2, 2)
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')

    plt.show()

def visualize_activations(nn):
    fig = plt.figure(figsize=(10, 10))

    i = 1
    for weights_for_neuron in nn.weights[0]:
        img = []
        for weight in weights_for_neuron[:-1]:
            img.append(np.sqrt(np.sum(weight * weight)))
        img = np.array(img).reshape((28, 28))
        ax = fig.add_subplot(10, nn.weights[0].shape[0] / 10, i)
        plt.imshow(img, cmap='gray')
        i = i+1
    plt.show()

def visualize_results(results):
    fig = plt.figure(figsize=(10, 10))

    i = 1
    for image in results:
        img = np.array(image).reshape((28, 28))
        ax = fig.add_subplot(1, 5, i)
        plt.imshow(img, cmap='gray')
        i = i + 1
    plt.show()


def add_noise(X):
    noise = np.random.normal(0.3, 0.1, 28 * 28)
    return np.clip(X + noise, 0, 1)

# The main script creates a network that learns MNIST images
if __name__ == '__main__':
    import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = np.array(training_data)
    X = training_data[:, 0]
    X = np.array(list(map(lambda x: x.flatten(), X)))

    nn = SparseAutoencoder()
    nn.input = np.array(list(zip(add_noise(X), X)))
    nn.add_layer(50)
    nn.add_layer(784)
    try:
        nn.learn(50, 0.01, 256, activation='relu')
    except Exception as e:
        print(e)

    visualize_activations(nn)

    res_images = []
    np.random.shuffle(test_data)
    for image in test_data[:30]:
        img = np.array([image[0].flatten()])
        img = add_noise(img)
        res = nn.classify(img)
        res_images.append(res)
        show_image((img, image[1]), res)


