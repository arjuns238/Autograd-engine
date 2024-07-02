from graphviz import Digraph
import matplotlib.pyplot as plt
import math
from graph_utils import draw_dot

class Value:
    def __init__(self, data, children = set(), op = '', label = ''):
        self.data = data
        self._prev = children
        self._op = op
        self.label = label
        self._backward = lambda: None
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data} label = {self.label})"
        
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, children = {self, other}, op = '+')
        def _backward():
            # Chain rule (local derivative dL/d_self = 1, global deriv = dl/d_out)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
        
    def __radd__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other + self        
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, children = {self, other}, op = '*')
        
        def _backward():
            # Chain rule (local derivative dL/d_self = other, global deriv = dl/d_out)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other * self

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data - other.data, children = {self, other}, op = '-')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        return out 

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other - self
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, children = {self,}, op = f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        if self.data <= 0:
            self.data = 1e-7
        out = Value(math.log(self.data), children = {self, }, op = 'log')

        def _backward():
            self.grad += 1.0/(self.data) * out.grad
        out._backward = _backward
        return out 

    def sigmoid(self):
        num = 1.0 / (1.0 + math.exp(-1.0*self.data))
        out = Value(num, children = {self, }, op = 'sigmoid')

        def _backward():
            self.grad += num * (1 - num) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        num = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(num, children = {self, }, op = 'tanh')

        def _backward():
            self.grad += (1.0 - num**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = self.data if self.data > 0 else 0
        out = Value(out, children = {self, }, op = 'ReLU')

        def _backward():
            deriv = 1 if self.data > 0 else 0 
            self.grad += deriv * out.grad
        out._backward = _backward

        return out

    def backward(self):
        top_order = []
        visited = set()
        def top_sort(root):
            if root not in visited:
                visited.add(root)
                for element in root._prev:
                    top_sort(element)
                top_order.append(root)
        top_sort(self)
        self.grad = 1.0
        for node in reversed(top_order):
            node._backward()