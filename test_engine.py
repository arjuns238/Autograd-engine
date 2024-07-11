from autograd import Value
from graph_utils import draw_dot
import torch
from nn import Linear, MLP


def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_ag, y_ag = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert y_ag.data == ypt.data.item()
    # backward pass went well
    assert x_ag.grad == xpt.grad.item()
    print("All test cases passed!!")


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size, activation='relu')
        # self.fc2 = Linear(hidden_size, hidden_size, activation='relu')
        # self.fc3 = Linear(hidden_size, hidden_size, activation='relu')
        self.fc4 = Linear(hidden_size, output_size, activation='sigmoid')
        self.layers = [self.fc1, self.fc4]
    def __call__(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        return x[0] if len(x) == 1 else x
    def parameters(self):
        p = []
        for l in self.layers:
            p.extend(l.parameters())
        return p
