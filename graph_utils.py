from graphviz import Digraph

# Create graph
def create_graph(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# Visualizing the graph
def draw_dot(root, image_format = "png", rankdir = "LR"):
    nodes, edges = create_graph(root)
    dot = Digraph(format = image_format, graph_attr = {"rankdir": rankdir})
    for node in nodes:
        # Creating a unique identifier for each node
        uid = str(id(node))
        dot.node(uid, label = f'{node.label} | Data: {node.data:.4f} | Grad: {node.grad: .4f}', shape = "record")
        # If the node is the result of an operation then connect the operation to the node
        if node._op:
            uid_op = uid+node._op
            dot.node(uid_op, label = node._op)
            dot.edge(uid_op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot   

