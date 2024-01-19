import random
import math


class Value:
    """A numerical value with backwards passing and gradient computation.

    Attributes:
        data (float): the wrapped numerical value.
        grad (float): the gradient of the value.
        label (str): a custom label for the object.
    """

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
        other = other if isinstance(other, Value) else Value(other)  # Wrap numbers with Value
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():  # Addition: gradients stay the same
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # Fallback method for num * Value
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float"
        out = Value(self.data ** other, (self,), f"^{other}")

        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __lt__(self, other):
        return self.data < other.data

    def __gt__(self, other):
        return self.data > other.data

    def __le__(self, other):
        return self.data <= other.data

    def __ge__(self, other):
        return self.data >= other.data

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return id(self)

    def __ne__(self, other):
        return self.data != other.data

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        try:
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        except OverflowError:
            if x > 0:
                t = 1.0
            else:
                t = -1.0
        '''
        if x < 0:
            t = 2 * math.exp(x) / (1+math.exp(2*x)) - 1
        else:
            t = 2 * (1 / math.exp(-2*x)) - 1
        '''
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Traces backward and computes gradients of all child nodes.
        """
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


class Neuron:
    """A single neuron, with inputs and output.

    Attributes:
        w (float): The weights of the neuron.
        b (float): The bias of the neuron.

    """
    def __init__(self, nin, linear=False):
        # Initialize with random weights and biases
        self.w = [Value(random.uniform(-1, 1), label="w") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label="b")
        self.linear = linear

    def __call__(self, x):
        # Weights and biases
        act = sum([wi*xi for wi, xi in list(zip(self.w, x))], self.b)
        act.label = "act"
        if self.linear:
            return act
        return act.tanh()

    def get_parameters(self):
        """
        :obj:`list` of :obj:`Value`: A list of all weights and biases (as Values) of the `Neuron`.
        """
        return self.w + [self.b]


class Layer:
    """A layer in the neural network, with inputs and outputs.

    Attributes:
        neurons (:obj:`list` of :obj:`neuron`): An list of the layer's neurons.
    """
    def __init__(self, nin, nout, linear=False):
        self.neurons = [Neuron(nin, linear) for i in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]  # List neuron forward-passes
        return outs[0] if len(outs) == 1 else outs

    def get_parameters(self):
        """
        :obj:`list` of :obj:`Value`: A list of all weights and biases (as Values) in the `Layer`, in no particular order.
        """
        params = []
        for neuron in self.neurons:
            ps = neuron.get_parameters()
            params.extend(ps)
        return params


class MLP:
    """A multi-layer perceptron with input, output, and  some optimization functions.

    Attributes:
        layers (:obj:`list` of :obj:`layer`): An ordered list of the MLP's layers.
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], i in range(len(nouts)-1, len(nouts)+1)) for i in range(len(nouts))]  # Create all layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_parameters(self):
        """
        :obj:`list` of :obj:`Value`: A list of all weights and biases (as Values) in the `MLP`, in no particular order.
        """
        params = []
        for layer in self.layers:
            ps = layer.get_parameters()
            params.extend(ps)
        return params

    def zero_grad(self):
        """
        Resets all gradients in the `MLP`.
        """
        for p in self.get_parameters():
            p.grad = 0.0  # Reset gradients to 0

    def nudge(self, step_size):
        """Nudges the `MLP` in the opposite direction of the gradient.

        Args:
            step_size: The step size multiplier of optimization.

        """
        for p in self.get_parameters():
            p.data -= step_size * p.grad  # nudge in opposite direction of gradient


# Utility functions
def _optim_sum(nums):
    """Recursively sums all elements of `nums` in pairs to minimize operation depth.

    Args:
        nums (:obj:`list` of :obj:`float`): the list of numbers to sum.

    Returns:
        The sum of all elements in `nums`.
    """
    if len(nums) == 2:
        return nums[0]+nums[1]
    if len(nums) == 1:
        return nums[0]
    return _optim_sum(nums[len(nums)//2:]) + _optim_sum(nums[:len(nums)//2])


def mean_squared_error(ys, ypred):
    """Computes mean squared error (MSE).

        Args:
            ys: ground-truth result
            ypred: predicted result

        Returns:
            float: the mean squared error between `ys` and `ypred`.
    """
    return _optim_sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])/len(ys)


def mean_abs_error(ys, ypred):
    """Computes mean absolute error (MAE).

    Args:
        ys: ground-truth result
        ypred: predicted result

    Returns:
        the mean absolute error between `ys` and `ypred`.
    """
    return sum([abs((yout - ygt).data) for ygt, yout in zip(ys, ypred)]) / len(ys)


def one_hot(string, classes):
    """Performs one-hot encoding of `string` based on possible `classes`.

    Args:
        string (str): the item to encode.
        classes (:obj:`list` of :obj:`str`): a list of all possible classes.
    Returns:
        :obj:`list` of :obj:`float`: a list of values, either -1.0 or 1.0, representing the one-hot encoding of `string`.
    """
    return [1.0 if i == string else -1.0 for i in classes]

