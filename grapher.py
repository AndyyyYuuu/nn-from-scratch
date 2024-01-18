from graphviz import Digraph, Source


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v, depth):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                '''
                if depth > 50:
                    v.label += " !2"
                    child.label += " !1"
                    break
                '''

                build(child, depth+1)


    build(root, 0)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)

            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    print("Rendering ... ")
    dot.render("graph.gv", view=True)
    print("Graph rendered!")
    return dot
