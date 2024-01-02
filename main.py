import math
import numpy
from matplotlib import pyplot
import util

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._backward = lambda: None
        self.label = label
        self.grad = 0
        self._prev = _children
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward(): # Addition: gradients stay the same
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        # Topological sort recursively
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # Build topological sort starting from this value
        build_topo(self)

        self.grad = 1.0
        # Backpropagate all nodes
        for node in reversed(topo):
            node._backward()

def main():
    a = Value(0.5, label="a")
    b = Value(0.3, label="b")
    c = Value(0.6, label="c")
    d = b*a; d.label = "d"
    e = b*c; e.label = "e"
    l = (d+e).tanh(); l.label = "l"
    l.grad = 1.0
    l.backward()
    util.draw_dot(l)

main()
'''
def derivative(f, x, h = 0.00000001):
    return numpy.round((f(x + h) - f(x))/h, int(abs(math.log(h, 10))-1))

print(derivative(lambda x: x**2+2*x+5, 0.430994))
x = numpy.linspace(-2, 2, 100)
y = derivative(lambda x: 3*x**4-4*x**3+x**2+2*x+5, x)
fig = pyplot.figure(figsize=(10, 5))
pyplot.plot(x, y)

pyplot.show()
'''