import numpy as np
import random
import networkx as nx
from graph import *


# Inspired by Karpathy's minigrad

class Term:
    def __init__(self, value, _in=(), _op=''):
        self.value = value
        self.grad = 0

        self._in = set(_in)
        self._op = _op
        self._back = lambda: None

    def __repr__(self) -> str:
        return f'Term(Value: {self.value})'

    def zero_grad(self) -> None:
        self.grad = 0

    def __add__(self, other):
        other = other if isinstance(other, Term) else Term(other)
        ret =  Term(self.value + other.value, (self, other), '+')

        def _back():
            self.grad += ret.grad
            other.grad += ret.grad
        ret._back = _back

        return ret
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Term) else Term(other)
        ret =  Term(self.value * other.value, (self, other), '*')

        def _back():
            self.grad += ret.grad * other.value
            other.grad += ret.grad * self.value
        ret._back = _back

        return ret
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, Term) else Term(other)
        ret =  Term(self.value ** other.value, (self, other), '^')

        def _back():
            self.grad += (other * self.data**(other-1)) * ret.grad
        ret._back = _back

        return ret

    def back(self):
        vis = set()
        top = []
        q = [self]
        while len(q) > 0:
            val = q.pop()
            top.append(val)
            vis.add(val)
            for c in val._in:
                if c not in vis:
                    q.append(c)
        
        self.grad = 1
        for n in top:
            n._back()
    
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __truediv__(self, other):
        return self * other**-1
    def __rtruediv__(self, other):
        return other * self**-1

class Layer:
    def __init__(self, n, n_in) -> None:
        self.neurons = [Neuron(n_in) for i in range(n)]

    def __repr__(self) -> str:
        return f'Layer(Neurons: {self.neurons})'

class Neuron:
    def __init__(self, n_in) -> None:
        self.w = [Term(random.uniform(-1, 1)) for i in range (n_in)]
        self.b = Term(0)
    
    def __repr__(self) -> str:
        return f'Neuron(w: {self.w}, b: {self.b})'




inputs = [[Term(i)] for i in range(10)]
inputs.append([Term(1)])
inputs = np.array(inputs)

l1 = [[Term(i * j) for j in range(11)] for i in range(5)]

l2 = [[Term(i) for i in range (5)]]

res = l2 @ (l1 @ inputs)

res[0][0].back()
graph_e(res[0][0])