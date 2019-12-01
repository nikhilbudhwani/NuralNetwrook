from NeuralNetwork import NeuralNewtowrk
import gzip
import shutil
from  ActivationFunction import Activation as af
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
class test:

    
    def __init__(self,network=None):
        file_content = None

        #setup
        if(network==None):
            self.network = NeuralNewtowrk()
            self.network.add_input_layer(784)
            self.network.add_hidden_layer(64,af.Sigmoid)
            self.network.add_hidden_layer(32,af.Sigmoid)
            self.network.add_output_layer([0,1,2,3,4,5,6,7,8,9],af.Sigmoid )
        else:
            self.network=network

        self.traning_file="test-data/number/train.csv"
        self.test_file="test-data/number/test.csv"
    
    #training
    def training(self,learning_rate):
        with open(self.traning_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            accuracy=0
            self.network.learning_rate=learning_rate
            for row in csv_reader:
                if(line_count==0):
                    line_count+=1
                    continue
                # if(line_count%2000==0):
                #     stop=input()
                accuracy=self.train([val/255 for val in map(int, row[1:])],int(row[0]))
                print(f'Processed first {line_count}  Traning Data with errors of {accuracy}.\n')
                # if(line_count%5==0):
                #     print(input())
                line_count+=1
        self.network.save_model("digit_mnn")

    def train(self,input:[],output):    
        return self.network.training(input,output)

    def predicte(self,input:[]):
        try:
            return self.network.predicte(input)
        except:
            self.save_model() 
        

    #testing
    def Testing(self):
        with open(self.test_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            more_inputs=None
            for row in csv_reader:
                try:
                    if(line_count==0):
                        line_count+=1
                        continue
                    inp_values=[val/255 for val in map(int, row[0:])]
                    rtn=self.predicte(inp_values)
                    plt.imshow(np.array(inp_values).reshape(28,28),cmap=plt.cm.binary)
                    print(f'Processed test number {line_count} reconize this as {rtn}')
                    plt.show()   
                    print(f'\nif this is correct then press 1 else 0:')
                    inp=input()
                    if(inp!='1'):
                        print(f'\n Please provide the correct in put from 0-9:')
                        val=input()
                    self.train(inp_values,int(val))
                    if(more_inputs=='2'):
                        line_count+=1
                        continue
                    print(f'\nplease press 1 to continue and 0 to exit and 2 to not ask again:')
                    more_inputs=input()
                    if(more_inputs=='0'):
                        break
                except:
                    print("Warning something went wrong")

                line_count+=1


               
def traninig(t):
    print("\nplease enter Number of epoc:")
    epoc=input()
    print(f"\nTraning Started.......... \n")    
    for i in range(1,int(epoc)+1):
        t.training(0.1)
    print(f"Traning Complete.......... \n")

def load_file():
    print(f"\nLoding model.......... \n")
    t=test(NeuralNewtowrk().get_model("digit_mnn"))
    print("model is loaded.")
    print("please select option: \n1.Train more\n2.Testing")
    inp=input()
    if(inp=='1'):
        traninig(t)
    else:
        testing(t)

def testing(t):
    print(f"Press enter for testing. \n")
    print(input())
    t.Testing()

if __name__ == "__main__":

    print("Please select the option \n1: Traning\n2:load the modal\nEnter the option:")
    t=None
    traing_or_save = input()
    if(traing_or_save=='1'):
        t=test()
        traninig(t)
        testing(t)
    else:
       load_file()


