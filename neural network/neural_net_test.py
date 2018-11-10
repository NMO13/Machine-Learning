import unittest
import numpy as np
from my_neural_network import MyNeuralNet

class TestNeuralNet(unittest.TestCase):

    # used for gradient checking
    # see https://www.youtube.com/watch?v=P6EtCVrvYPU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=54
    def calc_gradients(self, nn):
        epsilon = 0.00001
        def cost():
            c = 0.5 * (nn.output - nn.input[:,1]) ** 2
            return sum(c.flatten())

        grad_approx = [[[0 for k in range(len(nn.weights[i][j]))] for j in range(len(nn.weights[i]))] for i in range(len(nn.weights))]
        for layer in range(len(nn.weights)):
            for w0 in range(len(nn.weights[layer])):
                for w1 in range(len(nn.weights[layer][w0])):
                    value = nn.weights[layer][w0][w1]
                    nn.weights[layer][w0][w1] = value + epsilon
                    nn.feed_forward(nn.input[:,0])
                    c0 = cost()
                    nn.weights[layer][w0][w1] = value - epsilon
                    nn.feed_forward(nn.input[:,0])
                    c1 = cost()
                    res = c0 - c1
                    grad_approx[layer][w0][w1] = res / (2*epsilon)
                    nn.weights[layer][w0][w1] = value
        return grad_approx

    def test_raise_no_layers(self):
        nn = MyNeuralNet()
        with self.assertRaises(ValueError):
            nn.learn(1, 0.3, 1)
            nn.last_layer_neuron_count


    def test_raise_illegal_output(self):
        X = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]])
        y = np.array([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])
        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))

        nn.add_layer(1, 0)
        with self.assertRaises(ValueError):
            nn.learn(1, 1, 1)

    def test_simple_net(self):
        X = np.array([[0.05]])
        y = np.array([[0.01]])

        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))
        nn.add_layer(1, 0)
        nn.learn(1500, 0.3, 1)
        self.assertEqual(len(nn.weights), 1)
        self.assertEqual(nn.weights[0].shape, (1, 2))
        self.assertEqual(nn.weights[0][0][0], 1.5807061550409101)

    def test_backprop_1_iteration(self):
        X = np.array([[0.05, 0.10]])
        y = np.array([[0.01, 0.99]])

        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))
        nn.setup_test_conf()
        nn.learn(1, 0.5, 1)
        self.assertEqual(nn.weights[0][0][0], 0.14978071613276281)
        self.assertEqual(nn.weights[0][0][1], 0.19956143226552567)
        self.assertEqual(nn.weights[0][1][0], 0.24975114363236958)
        self.assertEqual(nn.weights[0][1][1], 0.29950228726473915)

        self.assertEqual(nn.weights[1][0][0], 0.35891647971788465)
        self.assertEqual(nn.weights[1][0][1], 0.4086661860762334)
        self.assertEqual(nn.weights[1][1][0], 0.5113012702387375)
        self.assertEqual(nn.weights[1][1][1], 0.5613701211079891)

    def test_backprop_1500_iteration(self):
        X = np.array([[0.05, 0.10]])
        y = np.array([[0.01, 0.99]])

        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))
        nn.setup_test_conf()
        nn.learn(1500, 0.5, 1)
        self.assertEqual(nn.weights[0][0][0], 0.1747576240731027)
        self.assertEqual(nn.weights[0][0][1], 0.24951524814620638)
        self.assertEqual(nn.weights[0][1][0], 0.2740253089080296)
        self.assertEqual(nn.weights[0][1][1], 0.3480506178160598)
        self.assertEqual(nn.output.shape, (1,2))

    def test_multiple_input_1_neuron_1_layer(self):
        X = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]])
        y = np.array([[0], [1], [1], [0], [0]])
        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))

        nn.add_layer(4, 0)
        nn.add_layer(1, 0)
        nn.learn(1500, 1, 5)
        res = nn.classify([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]])
        self.assertEqual(list(map(lambda x: int(round(x[0])), res)), [0, 1, 1, 0, 0])

    def test_multiple_input_2_neuron_1_layer(self):
        X = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]])
        y = np.array([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])
        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))
        nn.add_layer(4, 0)
        nn.add_layer(2, 0)
        nn.learn(1500, 1, 5)
        res = nn.classify([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]])
        self.assertEqual(list(map(lambda x: [int(round(x[0])), int(round(x[1]))], res)), [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])


    def test_multiple_input_50_neuron_2_layer(self):
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1],
                      [0, 0]])
        y = np.array([[0], [1], [1], [0], [0]])
        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))

        nn.add_layer(4, 0)
        nn.add_layer(1, 0)
        nn.learn(1500, 1, 5)
        res = nn.classify([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1],
                      [0, 0]])
        self.assertEqual(list(map(lambda x: int(round(x[0])), res)), [0, 1, 1, 0, 0])

    def test_gradient_checking(self):
        X = np.array([[0.05, 0.10]])
        y = np.array([[0.01, 0.99]])

        nn = MyNeuralNet()
        nn.activation = 'relu'
        nn.input = np.array(list(zip(X, y)))
        nn.setup_test_conf()
        grad_approx = np.array(self.calc_gradients(nn))
        nn.feed_forward(nn.input[:,0])
        grad = np.array(nn.backward(nn.input[:,1]))
        self.assertTrue(np.allclose(grad, grad_approx))

    def test_classify(self):
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1],
                      [0, 0]])
        y = np.array([[0], [1], [1], [0], [0]])
        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))

        nn.add_layer(4, 0)
        nn.add_layer(1, 0)
        nn.learn(1500, 10, 1)
        res = nn.classify(np.array([[0, 0], [1, 0]]))
        self.assertEqual(int(round(res[0][0])), 0)
        self.assertEqual(int(round(res[1][0])), 1)

    def test_relu(self):
        # XOR
        X = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        y = np.array([[0, 1],
                         [1, 0],
                         [1, 0],
                         [0, 1]])

        nn = MyNeuralNet()
        nn.input = np.array(list(zip(X, y)))

        nn.add_layer(10)
        nn.add_layer(10)
        nn.add_layer(2)
        nn.learn(1500, 10, 1, activation='relu')
        print(nn.classify(X))

if __name__ == '__main__':
    unittest.main()