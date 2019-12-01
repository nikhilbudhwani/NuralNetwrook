import numpy as np
from ActivationFunction import ActivateionClass

class Neuron():

    self.weights = None
    self.Outputvalue = 0
    self.output = None
    self.wtDotIn = 0

    def __init__(self, activationClass:):
        self.activationClass = activationFunction
        initilizeWeightsForInputLayer()

    def __init__(self, numberOfInputs: int, activationClass:ActivateionClass):
        self.weights = [numberOfInputs]
        self.activationClass = activationFunction
        intilizeWeights(numberOfInputs)

    def __init__(self, numberOfInputs: int, output: object, activationClass:ActivateionClass):
        self.weights = [numberOfInputs]
        self.activationClass = activationFunction
        intilizeWeights(numberOfInputs)
        self.output = output

    def initilizeWeightsForInputLayer(self):
        self.weights.Add(1)

    def intilizeWeights(self, numberOfInputs: int):
        for x in range(numberOfInputs):
            self.weights[x] = np.random.Generator().uniform(0, 1)

    def activate(self, inputs: []):
        rtn = 0
        for x, y in zip(inputs, self.weights):
            rtn = +x*y
        self.wtDotIn = rtn
        self.Outputvalue = self.activationClass.activate(rtn)
        return self.Outputvalue

    def differentiate(self):
        self.activationClass.differentiate(self.wtDotIn)
