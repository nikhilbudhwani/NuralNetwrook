import enum
import ActivationFunction as af
import numpy as np
from random import random
# seed random number generator
class Layer():

    def __init__(self):
        self.neurons = 0
        self.is_input_layer = False #if the layer is an input layer this can be segreated an can be abstract
        self.is_output_layer = False #if the layer is and output layer this can be segerated an can be abstract
        self.weight = None #are the input weights
        self.output = [] # are the original output object of the output layer
        self.output_values = [] #are the calculated output of each layer after the activation funtion
        self.activation_class = None #activation fuction of the neurons
        self.input_values = [] #total input of a neron it is multiply of self.weights*inputs from last payer
        self.input_from_last=[]#only input array
        self.bias=1.0
    #function to defin an input layer
    def define_input_layer(self, number_of_inputs: int):
        self.is_input_layer = True
        self.neurons = number_of_inputs
        self.input_values = []*self.neurons
        self.output_values = []*self.neurons
        self.activation_class = af.Identity()
        return self

    #function to defin and hidden layer
    def define_hidden_layer(self, activation_function: af.Activation, number_of_weights: int, number_of_neuron: int):
        self.activation_class = af.get_activation_class(activation_function)
        self.neurons = number_of_neuron
        self.output_values = []*self.neurons
        self.input_values = []*self.neurons
        self.weight = np.array(self.assign_weights(self.neurons, number_of_weights))
        self.bias=random()
        return self

    #function to define an output layer
    def define_output_layer(self, activation_function: af.Activation, number_of_weights: int, outputs: []):
        assert len(outputs) > 0, "output should have atlest one Value"
        self.is_output_layer = True
        self.output = outputs
        self.neurons = len(outputs)
        self.output_values = []*self.neurons
        self.input_values = []*self.neurons
        self.weight = np.array(self.assign_weights(self.neurons, number_of_weights))
        self.activation_class = af.get_activation_class(activation_function)
        return self
    
    # initilization of the weights to random
    def assign_weights(self, row, col):
        rtn = []
        for r in range(row):
            rtnc = []
            for c in range(col):
                r=random() 
                #r=0.0
                rtnc.append(r)
            rtn.append(rtnc)
        return rtn

    def feed(self, inputs: []):
        self.input_from_last = np.array(inputs, dtype=np.float64)
        self.input_values = np.dot(self.weight, self.input_from_last.T)
        out = self.activation_class.activate(self.input_values+self.bias)
        self.output_values = np.array(out, dtype=np.float64)
        return self.output_values

    def feed_input(self, inputs):
        self.input_from_last =self.input_values = inputs
        out = self.activation_class.activate(self.input_values)
        self.output_values = np.array(out, dtype=np.float64)
        return self.output_values

    def update_weight(self, learning_rate, error: []):
        delta = error * np.array(self.activation_class.differentiate(self.output_values) , dtype=np.float64)
        error_for_last_layer= np.dot(delta, self.weight)
        change = np.dot(self.input_values.T,delta)
        self.weight += learning_rate * change
        self.bias+=learning_rate*change
        return error_for_last_layer
