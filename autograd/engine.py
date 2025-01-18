class Value:
    def __init__(self, value, _children=()):
        self.data = value
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out 
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        visited = set()
        graph = []

        def topological_sort(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sort(child)
                graph.append(v)
        topological_sort(self)

        self.grad = 1
        for v in graph.reverse():
            v._backward()
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return -1 * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __rtruediv__(self, other):
        return other * self ** -1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"