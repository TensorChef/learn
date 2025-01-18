from engine import Value
import numpy as np

class Module:
    
    def parameters(self,):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):

    def __init__(self, input_size, non_linearity=True):
        self.W = [Value(np.random.uniform(-1, 1)) for _ in range(input_size)]
        self.b = Value(0)
        self.non_linearity = non_linearity

    def __call__(self, x):
        act = sum((w_i * x_i for w_i, x_i in zip(self.W, x)), self.b)
        return act.relu() if self.non_linearity else act
    
    def parameters(self):
        return self.W + [self.b]
    
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self._neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, X):
        out = [n(X) for n in self._neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self._neurons for p in n.parameters()]
    
class Perceptron(Module):

    def __init__(self, nin: int, nouts : list):
        sizes = [nin] + nouts
        self._layers = [Layer(sizes[i], sizes[i+1], non_linearity=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self._layers for p in layer.parameters()]