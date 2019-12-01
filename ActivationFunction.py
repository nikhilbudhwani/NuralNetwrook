import numpy as np
import abc
import enum
from scipy.special import expit
import scipy


class ActivateionClass(abc.ABC):
    @abc.abstractmethod
    def activate(self, value):
        pass

    def differentiate(self, value):
        pass


class Relu(ActivateionClass):
    def activate(self, value):
        return np.maximum(0, value)

    def differentiate(self, value):
       return  np.where(value <= 0, 0, 1)


class Sigmoid(ActivateionClass):
    def activate(self, value):
        return 1 / (1 + (expit(- value)))

    def differentiate(self, value):
        return value*(1-value)


class Identity(ActivateionClass):
    def activate(self, value):
        return value

    def differentiate(self, value):
        return 0

class Step(ActivateionClass):
    def activate(self, value):
         return [1 if(val>0) else -1 for val in value]
    
    def differentiate(self, value):
        return [0 for val in value]
        


class Softmax(ActivateionClass):
    def activate(self, value):
       return scipy.special.softmax(value)
        

    def differentiate(self, value):
       return scipy.special.logsumexp(value)


class Activation(enum.Enum):
    Sigmoid = 1,
    Relu = 2,
    Identity = 3,
    Softmax = 4,
    Step=5


__switcher = {
    Activation.Identity: Identity(),
    Activation.Relu: Relu(),
    Activation.Sigmoid: Sigmoid(),
    Activation.Softmax: Softmax(),
    Activation.Step:Step()
}


def get_activation_class(activation: Activation):
    return __switcher.get(activation)
