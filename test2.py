from NeuralNetwork import NeuralNewtowrk as nn
import ActivationFunction as af


class testXOR():
    def __init__(self):
        net = nn()
        net.add_input_layer(2)
        net.add_hidden_layer(2, af.Activation.Sigmoid)
        net.add_output_layer([0.1], af.Activation.Sigmoid)
        inp = [[0.4, -0.7], [0.3, -0.5], [0.6, 0.1], [0.2, 0.4],[0.1,-0.2]]
        out = [0.1, 0.05, 0.3, 0.25,0.12]
        for _ in range(1, 100):
            for i, o in zip(inp, out):
                net.training(i, o)
                wt=[ l.weight for l in net.newtwork]
                print(wt)
                print("\n")
        print("Traning complete for testing pleass enter")
        a = input()
        for ti in inp:
            print(ti)
            print( "oupt is " + str(net.predicte(ti)))
            b = input()
        
        

if __name__ == "__main__":
    t= testXOR()