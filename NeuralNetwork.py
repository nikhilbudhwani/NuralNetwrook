from Layer import Layer
import numpy as np
from ActivationFunction import Activation
import pickle 
import enum


class NeuralNewtowrk:

   # self.WeightsArray:[]
    def __init__(self):
        self.newtwork = []
        self.learning_rate = 0.25
        self.weightInit=weightInit.Random

    def add_input_layer(self, numberOfInputs: int):
        if(len(self.newtwork) > 0):
            raise Exception("Please inizilize Neural Network")
        layer=Layer().define_input_layer(numberOfInputs)
        self.newtwork.append(layer)

    def add_hidden_layer(self, number_of_neuron: int, activation_function: Activation):
        last_layer = self.newtwork[-1]
        if(self.newtwork.count == 0):
            self.add_input_layer(number_of_neuron)
        elif(last_layer.is_output_layer):
            self.newtwork.insert(-2, Layer().define_hidden_layer(activation_function, last_layer.neurons, number_of_neuron))
        else:
            self.newtwork.append(Layer().define_hidden_layer(activation_function,last_layer.neurons, number_of_neuron))  

    def add_output_layer(self, output_value: [], activation_function: Activation):
        last_layer = self.newtwork[-1]
        self.newtwork.append(
            Layer().define_output_layer( activation_function,last_layer.neurons, output_value))

    def training(self, inputs: [], output):
        calculeOuptput = self.forward_feed(inputs)
        expectedOutPut = [(1 if out == output else 0)
                          for out in self.newtwork[-1].output]
        predition = self.backward_propogation(calculeOuptput, expectedOutPut)
        return predition

    def backward_propogation(self, predicted_output, actual_output):
        out_error = error = actual_output-predicted_output

        for x in range(len(self.newtwork)-1,0, -1):
            error = self.newtwork[x].update_weight(self.learning_rate, error)
    
        mse = sum([out**2 for out in out_error])
        loss= np.mean(np.square(out_error))
        return loss

    def forward_feed(self, inputs: []):
        out = self.newtwork[0].feed_input(np.array(inputs))
        for layer in range(1, len(self.newtwork)):
            out = self.newtwork[layer].feed(out)
        return out
    
    def predicte(self,input:[]):
        out=self.forward_feed(input)
        calculate_output=max(out)
        index_value= list(out).index(calculate_output)
        calculate_out_value=self.newtwork[-1].output[index_value]     
        return calculate_out_value                 
        # if(output!=calculate_out):
        #     expectedOutPut = [(1 if out == output else 0)
        #                   for out in self.newtwork[-1].output] 
        #     self.backward_propogation(out,exceptedOutput)

        # return {success:output==calculate_out,probability:index_value}

    def get_model(self,file_name):
        with open("model/"+ file_name, "rb") as f:
            dump = pickle.load(f)
            self.newtwork = dump.newtwork
            self.learning_rate=dump.learning_rate
        return self

    def save_model(self,file_name):
        with open("model/" +file_name, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    

class weightInit(enum.Enum):
    Zero=0,
    Random=1




