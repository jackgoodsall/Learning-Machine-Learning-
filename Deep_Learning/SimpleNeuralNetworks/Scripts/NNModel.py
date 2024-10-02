import numpy as np
from Deep_Learning.SimpleNeuralNetworks.Scripts.NNLayerArchitecture import *
from NNFunctionUtils import * 

class NeuralNetwork():
    '''
    First attempt at making a neural network from scratch. At this point never worked with any NN modules so not sure of standard structure of 
    networks like in pytorch and tensorflow. 
    Used some help from google with things like array sizes to help it run with multiple training examples at the same time. Maybe should of started 
    with creating a simpler network first but that's boring. 
    
    '''
    def __init__(self) -> None:
        # Initalise the network
        self.layers : list[ NNLayers ] = []
        self.has_input_layer : bool = False
        self.has_outout_layer : bool = False
        
    
    def train(self, input_data: np.ndarray, target_values: np.ndarray, 
                epoches : int = 1000, learn_rate : float = 0.01, n_tol : float = 0.00001, 
                print_information : bool = False) -> None:
        '''
        Train the neural network based on the given input data, for now using whole training examples each time, which I'm pretty sure is both computationally ineffective and 
        worse for learning purposes (will change to batch once it actually works).
        input_data : np.ndarray
            shape(number_datapoints, n_features)
        target_data: np.ndarray
            shape(number_datapoints, 1)
        '''
        # Variables to compare costs (might change to save all costs later for plotting purposes but not important rn).
        last_cost = 0
        current_cost = 0
        for _ in range(epoches):
            
            # First forward pass the information
            outputs = self._forward_pass(input_data)

            # Calculate cost function
            current_cost = FunctionUtils.cost_function(outputs, target_values)

            # Back propagate
            self._back_propagation(target_values, learn_rate)
            # Check to see if the training cost has converged
            if abs(last_cost - current_cost) < n_tol:
                if print_information:
                    print("Network converged! Training finished")
                    print(f"Converged on a training cost of {current_cost} in {_} epoches")
                return
            
            last_cost = current_cost
        if print_information:
            print("Training finished")
            print(f"Did not converge in {epoches} epoches, consider changing the learn rate, increasing the tolerance or increasing the number of epoches.")
          
        
    def predict(self, predict_data: np.ndarray) -> np.ndarray:
        '''
        Predict values that are fed into the neural network by simpily forward passing the inputs
        '''
        return self._forward_pass(predict_data)
        
    
    def add_dense(self,  n_neurons : int, activation_function : bool = True) -> None | str:
        '''
        Method to add a dense layer to the network class, first checks that a input layer exists and that there isn't an output layer.
        Will eventualy change the activation_function = True to be able to select from a range of activation functions
        '''

        # Add a dense layer to the network
        if self.has_input_layer == False:
            return "Network does not have an input layer"
        elif self.has_outout_layer:
            return "Network has an output layer, can not add another Dense layer"
        else:   
            self.layers.append( DenseLayer(self.layers[-1].output_size, n_neurons, activation = activation_function) )            
        
    
    def add_input(self, input_size : int, ) -> None | str:
        '''
        Method for adding an input layer to the network, checks that there is no layers added already
        '''
        if self.layers != None:
            self.layers.append( InputLayer(input_size) )
            self.has_input_layer = True
        else:
            return "Network already has an input layer"


    def add_output(self, output_size:int) -> None | str:
        '''
        Method for adding output layer to the network, checks that it has an input layer and that it doesnt already have an output layer, should really check to make sure 
        a dense layer exists but who is creating an network that is just an input and output layer.
        '''
        # First check if there is already an output layer
        if self.has_input_layer == False:
            return "Network has no input layer"
        # Next check to make sure there is atleast an input layer
        elif self.has_outout_layer:
            return "Network already has an output layer"
        else:
            self.layers.append( OutputLayer(self.layers[-1].output_size, output_size) )
            self.has_outout_layer = True

            
    def _forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        '''
        Private method for forward passing through the network.
        '''
        output : np.ndarray = input_values
        for layer in self.layers:
            output = layer._forward(output)
        return output
        

    def _back_propagation(self, target_values : np.ndarray, learn_rate : float) -> None:
        '''
        Private method for back propagating through the network
        '''
        last_layer = self.layers[-1]
        kronker_current = 0
        kronker_last = last_layer.output - target_values 
        for index, layer in zip(range(len(self.layers)-1, 0, -1), self.layers[::-1]):
            
            if isinstance(layer, OutputLayer):
    
                layer.weights -= learn_rate *  layer.input.T @ kronker_last  / np.linalg.norm(layer.input.T @ kronker_last )

            else: 
                kronker_current = kronker_last @ last_layer.weights.T * FunctionUtils.relu_derivitive(layer.output_activation)

                weight_gradient = self.layers[index-1].output.T @ kronker_current
                
                bias_gradient = np.mean(kronker_current, axis= 0, keepdims= True)

                bias_gradient = np.clip(bias_gradient, -0.5, 0.5)

                weight_gradient = weight_gradient /  np.linalg.norm(weight_gradient)

                layer.weights -= learn_rate *  weight_gradient 

                layer.bias -= learn_rate * bias_gradient
               
                kronker_last = kronker_current
            

            last_layer = layer



 

                

   
    
    




    