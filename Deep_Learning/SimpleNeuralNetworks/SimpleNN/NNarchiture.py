import numpy as np
from NNFunctionUtils import * 


class NNLayers:
    '''
    Base class for NN layers to be inherited by layers to ensure proper utilities of a layer are implemented
    '''
    def __init__(self):
        return NotImplementedError
    
    def _forward(self):
        return NotImplementedError
    

class DenseLayer(NNLayers):
    '''
    Class for a dense layer in a neural network
    '''

    def __init__(self, n_inputs : int , n_neurons : int, activation = True, seed: int = 42) -> None:
        # Set seed for layer, helps in debuggin
        np.random.seed(seed)
        # Set input and output size attributes
        self.input_size : int = n_inputs
        self.output_size : int = n_neurons
        # Initalise the weights and bias of the dense layer
        self.weights : np.ndarray = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.bias : np.ndarray = np.zeros((1, n_neurons)) 
        '''
        If activation is true set activation function to relu
        This will be changed layer to accomidate more activation funciton choses like:
            - softmax
            - tanh
            - leaky relu
            - and more
        but just want to focus on basics for now
        '''
        if activation:
            self.activation_function = FunctionUtils.relu
            

    def _forward(self, input: np.ndarray) -> np.ndarray:
        '''
        Function for forward propagating through the layer, saves input, output and output with activaiton for 
        use in back propagation.
        '''

        self.output : np.ndarray = np.dot(input, self.weights) + self.bias
        self.input : np.ndarray = input
        # If activation is used
        # not sure if easier or better way to do this but doesnt matter for now
        try:
            self.output_activation = FunctionUtils.relu(self.output)
        except:
            self.output_activation = self.output
        return self.output_activation
           

class InputLayer(NNLayers):
    '''
    Class representing an input layer for a neural network
    '''
    
    def __init__(self, n_input : int) -> None:
        self.input_size :int = n_input
        self.output_size : int = n_input

    def _forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = input
        try:
            return input
        except: 
            return IndexError
        

class OutputLayer(NNLayers):
    '''
    Class representing an output layer for a neural network
    '''

    def __init__(self, n_input: int , n_output : int) -> None:
            self.input_size :int = n_output
            self.output_size : int = n_output
            self.weights : np.ndarray = np.random.randn(n_input, n_output) * np.sqrt(2 / n_input)
    
    
    def _forward(self, input: np.ndarray, activation_function : bool = False) -> np.ndarray:
        self.input = input
        self.output = input
        try:
            self.output =  np.dot(input, self.weights) 
        except: 
            pass
        return self.output