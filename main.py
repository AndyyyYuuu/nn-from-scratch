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
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), "tanh")
        print(x)
        return out



def ex():
    a = Value(5.0, label="a")
    b = Value(3.0, label="b")
    c = Value(6.0, label="c")
    d = b*a; d.label = "d"
    e = b*c; e.label = "e"
    l = (d+e).tanh(); l.label = "l"
    l.grad = 1.0
    util.draw_dot(l)
ex()

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