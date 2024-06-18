from autograd import Value
from graph_utils import draw_dot
import random

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def __call__(self, x):
        # x is (m, in_features)
        # w should be (in_features, out_features)
        assert isinstance(x, list)

        w = []
        for i in range(self.in_features):
            n_features = []
            for j in range(self.out_features):
                n_features.append(Value(random.uniform(-1, 1)))
            w.append(n_features)

        assert len(w) == self.in_features and len(w[0]) == self.out_features

        wx = [[Value(0) for _ in range(len(w[0]))] for _ in range(len(x))]
        print("wx shape", len(wx), len(wx[0]))

        # Perform matrix multiplication
        for i in range(len(x)):
            for j in range(len(w[0])):
                for k in range(len(w)):
                    wx[i][j] += x[i][k] * w[k][j]
        print("wx shape", len(wx), len(wx[0]))
        y = [[Value(0) for _ in range(len(wx[0]))] for _ in range(len(wx))]
        if self.bias:
            biases = [[Value(0) for _ in range(len(w[0]))] for _ in range(len(x))]
            for i in range(len(wx)):
                for j in range(len(wx[0])):
                    y[i][j] = wx[i][j] + biases[i][j]
        else:
            y = wx
        return y     

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})" 