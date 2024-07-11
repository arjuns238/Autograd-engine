from autograd import Value
from graph_utils import draw_dot
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, nin, activation = 'relu'):
        self.nin = nin
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation
    
    def __call__(self, x):
        act = 0.0
        for x1, w1 in zip(x, self.w):
            act += x1 * w1
        act += self.b
        if self.activation == 'relu':
            out = act.relu()
        elif self.activation == 'sigmoid':
            out = act.sigmoid()
        return out
    
    def parameters(self):
        return self.w + [self.b]


class Linear(Module):
    def __init__(self, in_features, out_features, activation = 'relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.neurons = [Neuron(in_features, activation = self.activation) for _ in range(out_features)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})" 
    
    def parameters(self):
        p = []
        for n in self.neurons:
            p.extend(n.parameters())
        return p 
