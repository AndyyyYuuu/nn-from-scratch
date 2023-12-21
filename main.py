import math
import numpy
from matplotlib import pyplot
import util

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.label = label
        self.grad = 0
        self._prev = _children
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")


a = Value(5.0)
b = Value(3.0)
c = Value(6.0)
util.draw_dot(a+b*c)
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