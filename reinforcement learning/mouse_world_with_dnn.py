from .mouse_world import MouseWorld
from ..neural_network.my_neural_network import MyNeuralNet

class MouseWorld2(MouseWorld):

    def __init__(self):
        self.net = MyNeuralNet()

    def move(self, eps):
        pass