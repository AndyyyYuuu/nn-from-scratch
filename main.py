import math
import random
import csv
import numpy
from matplotlib import pyplot
import util
weather_map = {
    0: "sun",
    1: "fog",
    2: "drizzle",
    3: "rain",
    4: "snow",
    "sun": 0,
    "fog": 1,
    "drizzle": 2,
    "rain": 3,
    "snow": 4
}

NUM_CLASSES = 5
STEP_SIZE = 0.01
NUM_EPOCHS = 100

all_x = []
all_y = []


def equalize_classes(data_list, num_classes):
    categories = [[] for _ in range(num_classes)]
    for row in data_list:
        categories[weather_map.get(row[5])].append(row)
    targ_len = min([len(c) for c in categories])
    out = []
    for sublist in [c[:targ_len] for c in categories]:
        out.extend(sublist)

    return out


with open('data/seattle-weather.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)
    print(len(data))
    data = equalize_classes(data, NUM_CLASSES)
    print(len(data))
    random.shuffle(data)
    for row in data:
        all_x.append([float(item) for item in row[1:5]])
        weather_index = weather_map[row[5]]
        all_y.append([1.0 if i == weather_index else -1.0 for i in range(NUM_CLASSES)])

train_x = all_x[:math.floor(len(all_x)*0.8)]
train_y = all_y[:math.floor(len(all_x)*0.8)]
valid_x = all_y[math.floor(len(all_x)*0.2):]
valid_y = all_y[math.floor(len(all_x)*0.2):]

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
    def __init__(self, nin):
        # Initialize with random weights and biases
        self.w = [Value(random.uniform(-1, 1), label="w") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label="b")

    def __call__(self, x):
        # Weights and biases
        act = sum([wi*xi for wi, xi in list(zip(self.w, x))], self.b)
        act.label = "act"
        out = act.tanh()
        return out

    def get_parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]  # List neuron forward-passes
        return outs[0] if len(outs) == 1 else outs

    def get_parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.get_parameters()
            params.extend(ps)
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]  # Create all layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.get_parameters()
            params.extend(ps)
        return params

def optim_sum(l):
    if len(l) == 2:
        return l[0]+l[1]
    if len(l) == 1:
        return l[0]
    return optim_sum(l[len(l)//2:]) + optim_sum(l[:len(l)//2])


def mean_squared_error_o(ys, ypred):
    print(ys, ypred)
    return sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

def mean_squared_error(ys, ypred):
    s = Value(0.0, label="sqerr")
    values = []
    for i in range(len(ys)):
        for j in range(len(ys[i])):
            values.append((ys[i][j]-ypred[i][j])**2)

    return optim_sum(values)
    # s = (ys[0][0]-ypred[0][0])**2+(ys[0][1]-ypred[0][1])**2

    return s
    # return sum([mean_squared_error_o(ygt, yout) for ygt, yout in zip(ys, ypred)])
    # return sum([(yout - ygt)**2 for sublist_gt, sublist_pred in zip(ys, ypred) for ygt, yout in zip(sublist_gt, sublist_pred)])# / sum(len(sublist_gt) for sublist_gt in ys)



nn = MLP(4, [2, 2, 5])
# util.draw_dot(nn([2.0, 3.0, -1.0]))


'''
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets


train_x = [
    [0.2, 0.3],
    [0.3, 0.1],
    [-0.3, 0.2],
    [0.5, 0.3],
    [-0.1, 0.9],
    [-0.5, -0.1]
]
'''
#train_y = [0.5, 0.5, -0.1, 0.8, 0.8, -0.6]



parameters = nn.get_parameters()
print(f"Parameters: {len(parameters)}")

# util.draw_dot(loss)

for i in range(100):
    print(f"\nEpoch: {i}")
    pred_y = [nn(x) for x in train_x]  # Forward pass
    #util.draw_dot(ypred[0][0])
    train_loss = mean_squared_error(train_y, pred_y)
    #loss = [mean_squared_error_o(i, j) for i, j in zip(train_y, ypred)]  # Compute loss


    print(f"Training Loss: {train_loss.data}")
    for p in nn.get_parameters():
        p.grad = 0.0  # Reset gradients to 0
    train_loss.backward()  # Backward pass
    # util.draw_dot(loss)
    for p in parameters:
        p.data -= STEP_SIZE * p.grad  # nudge in opposite direction of gradient
    STEP_SIZE *= 0.95
    print("Validating ...")
    #VALIDATION

    pred_y = [nn(x) for x in valid_x]
    valid_loss = mean_squared_error(train_y, pred_y)
    total = len(pred_y)
    correct = 0
    for p, y in zip(pred_y, valid_y):
        print(p.index(max(p)), end='')
        if p.index(max(p)) == y.index(max(y)):

            correct += 1

    print(f"\nAccuracy: {correct}/{total} = {round(correct/total*100,2)}%")
    print(f"Loss: {valid_loss.data}")

    for p in nn.get_parameters():
        p.grad = 0.0  # Reset gradients to 0