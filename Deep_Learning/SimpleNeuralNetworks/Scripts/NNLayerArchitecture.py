import numpy as np
from NNFunctionUtils import * 


class NNLayers:
    '''
    Base class for NN layers to be inherited by layers to ensure proper utilities of a layer are implemented, also includes
    atribute of activaiton functions.
    '''
    _activation_function_dict = {
        "ReLU" : ActivationFunctions.relu,
        "Softmax" : ActivationFunctions.softmax,
        "None" : False
    }


    def __init__(self):
        '''
        Ensures __init__ is implemented in layers
        '''
        return NotImplementedError
    

    def _forward(self):
        '''Ensures _forward is implemented in layers'''
        return NotImplementedError
    

    @property
    def _activation_function(self) -> ActivationFunctions:
        return self._activation_function
    

    @_activation_function.setter
    def _activation_function(self, activation_name : str) -> ActivationFunctions:
        ''' Setter for activaiton function, maps str to function in activation functions'''
        if activation_name in self._activation_function_dict.values():
            self._activation_function = self._activation_function_dict[activation_name]
        else:
            self._activation_function = self._activation_function_dict["None"]


class DenseLayer(NNLayers):
    '''
    Class for a dense layer in a neural network
    '''

    def __init__(self, n_inputs : int , n_neurons : int, activation : str = "", seed: int = 42) -> None:
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
        but just want to focus on basics for now.
        02/10/24
        Changed above to use getters and setters for easy use of multiple activation functions, saving above comment to keep track of progress.
        Default choice is None current implemented by returning False
        '''
        self._activation_function = activation
            

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
            self.output = self._activation_function(self.output)
        except:
            pass
        return self.output
           

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
    '''Class representing an output layer for a neural network'''

    def __init__(self, n_input: int , n_output : int, activation_function : str = "None") -> None:
            ##
            self.input_size : int = n_output
            self.output_size : int = n_output
            self.weights : np.ndarray = np.random.randn(n_input, n_output) * np.sqrt(2 / n_input)
            self._activation_function = activation_function
    
    def _forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output =  np.dot(input, self.weights) 
        try:
            self.output = self._activation_function(self.output)
        except: 
            pass
        return self.output