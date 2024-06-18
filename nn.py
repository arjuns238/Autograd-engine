from Autograd import Value
from graph_utils import draw_dot

x1 = Value(2.0, label = 'x1')
x2 = Value(6.0, label = 'x2')

w1 = Value(-3.0, label = 'w1')
w2 = Value(1.0, label = 'w2')

b = Value(0.6, label = 'b')
# x1w1 + x2w2 + b
x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
x1w1x2w2b = x1w1x2w2 + b; x1w1x2w2b.label = 'x1w1x2w2 + b'
o = x1w1x2w2b.relu(); o.label = 'o'

graph = draw_dot(o)
graph.view()