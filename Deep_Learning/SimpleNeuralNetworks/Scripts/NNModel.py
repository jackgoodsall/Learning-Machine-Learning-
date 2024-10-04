import numpy as np
from NNLayerArchitecture import *
from NNFunctionUtils import * 

class NeuralNetwork():
    '''
    First attempt at making a neural network from scratch. At this point never worked with any NN modules so not sure of standard structure of 
    networks like in pytorch and tensorflow. 
    Used some help from google with things like array sizes to help it run with multiple training examples at the same time.
    Class Atributes:
    _loss_functions_dict : dict
        dictionary mapping strings to loss functions from NNFunctionUtils script
    '''
    _loss_functions_dict= {
        "LSE" : CostFunctions.LSE,
        "BCEL" : CostFunctions.BCEL,
        "CEL" : CostFunctions.CEL
    }


    def __init__(self) -> None:
        # Initalise the network
        self.layers : list[ NNLayers ] = []
        self.has_input_layer : bool = False
        self.has_outout_layer : bool = False
        
    
    def train(self, input_data: np.ndarray, target_values: np.ndarray, 
                epoches : int = 1000, learn_rate : float = 0.01, n_tol : float = 0.00001, 
                print_information : bool = False, loss_function : str = "LSE" ) -> None:
        '''
        Train the neural network based on the given input data, for now using whole training examples each time, which I'm pretty sure is both computationally ineffective and 
        worse for learning purposes (will change to batch once it actually works).
        input_data : np.ndarray
            shape(number_datapoints, n_features)
        target_data: np.ndarray
            shape(number_datapoints, n_labels)
        '''
        # Use setter for loss function (converts the str into a function)
        self._loss_function = loss_function
        # Variables to compare costs (might change to save all costs later for plotting purposes but not important rn).
        # 02/10 - Changed to list
        _loss_function = self._loss_functions_dict[self._loss_function]
        costs = np.ndarray((epoches, 1))
        last_cost = 0
        for cur_iteration in range(epoches):
            
            # First forward pass the information
            outputs = self._forward_pass(input_data)

            # Calculate cost function
            current_cost = _loss_function(outputs, target_values)

            # Back propagate
            self._back_propagation(target_values, learn_rate)
            # Check to see if the training cost has converged
            if abs(last_cost - current_cost) < n_tol:
                if print_information:
                    print("Network converged! Training finished")
                    print(f"Converged on a training cost of {current_cost} in {cur_iteration} epoches")
                return
            
            last_cost = current_cost
            costs[cur_iteration] = current_cost
        if print_information:
            print("Training finished")
            print(f"Did not converge in {epoches} epoches, consider changing the learn rate, increasing the tolerance or increasing the number of epoches.")
          
        
    def predict(self, predict_data: np.ndarray) -> np.ndarray:
        '''Predict values that are fed into the neural network by simpily forward passing the inputs'''
        return self._forward_pass(predict_data)
        
    
    def add_dense(self,  n_neurons : int, activation_function : str = "") -> None | str:
        '''
        Method to add a dense layer to the network class, first checks that a input layer exists and that there isn't an output layer.

        Will eventualy change the activation_function = True to be able to select from a range of activation functions
        02/10 - Changed to use atributes/setters and getters to change activation to str and added more functions.
        '''
        # Add a dense layer to the network
        if self.has_input_layer == False:
            return "Network does not have an input layer"
        elif self.has_outout_layer:
            return "Network has an output layer, can not add another Dense layer"
        else:   
            self.layers.append( DenseLayer(self.layers[-1].output_size, n_neurons, activation = activation_function) )            
        
    
    def add_input(self, input_size : int, activation_function : str = "ReLU") -> None | str:
        '''Method for adding an input layer to the network, checks that there is no layers added already'''
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
        Private method for back propagating and optimising the paramaters in the neural network.
        '''
        # Starting with the last layer
        last_layer = self.layers[-1]
        # Set Current delta function to 0, so can use variable later
        deltas_list = []
        # Set Current delta equal to derivitve of our cost function
        
        _loss_function_derivitive = getattr(CostFunctionsDerivitives, self._loss_function)

        deltas_list.insert(0, _loss_function_derivitive(last_layer.output, target_values))

        # Loop backwards through the layer
        for index, layer in zip(range(len(self.layers)-1, 0, -1), self.layers[::-1]):
            
            # If layer is output then don't need to update bias, currently not supported for activation on output will change later.
            if isinstance(layer, OutputLayer):
                # W -> W - lr * X.T * delta
                layer.weights -= learn_rate *  layer.input.T @ deltas_list[-1]  / np.linalg.norm(layer.input.T @ deltas_list[-1] )
            # If any other layer need bias update
            else: 
                # Current delta function
                # 
                
                deltas_list.insert(0, deltas_list[0] @ last_layer.weights.T * layer.output_derivitive )
              
                weight_gradient = self.layers[index-1].output.T @ deltas_list[0]
                
                # Bias gradient is mean value of kronker along column, size (1, n_neurons)
                bias_gradient = np.mean(deltas_list[0], axis= 0, keepdims= True)
                # Clip gradient to stop them exploding
                bias_gradient = np.clip(bias_gradient, -0.5, 0.5)
                # Normalise weight gradients to stop them exploding
                weight_gradient = weight_gradient /  np.linalg.norm(weight_gradient)
                # Update layer and weights 
                layer.weights -= learn_rate *  weight_gradient 
                layer.bias -= learn_rate * bias_gradient
                # Set last kronker to current one
  
            # Set last layer to current one
            last_layer = layer


    @property
    def _loss_function(self) -> CostFunctions:
        # Getter for the loss function atribute
        return self._internal_loss_function
    

    @_loss_function.setter
    def _loss_function(self, name : str) -> None:
        '''Setter for the loss function, maps the string to the function in FunctionUtils'''
        if name in self._loss_functions_dict.keys():
            self._internal_loss_function = name
        else:
            self._internal_loss_function = "LSE"



 

                

   
    
    




    