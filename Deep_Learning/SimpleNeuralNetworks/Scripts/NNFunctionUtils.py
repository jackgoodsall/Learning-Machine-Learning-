import numpy as np

class CostFunctions:
    '''
    Class containing static methods of functions used in Neural networks
    '''

    @staticmethod
    def LSE_function(predicted_values: np.ndarray, expected_values : np.ndarray) -> np.ndarray:
        # Calculate LSE cost function between predicted and expected values

        numerator = np.sum((predicted_values - expected_values)**2, axis = 0)
        denominator  =  2 * predicted_values.shape[0]
        return numerator/ denominator
    

    @staticmethod
    def BCEL_function(predicted_label : np.ndarray, actual_label : np.ndarray) -> np.ndarray:
        # Static method for Binary Cross Entropy loss
        return NotImplementedError

    
    def CEL_function(predicted_labels : np.ndarray, actual_labels : np.ndarray) -> np.ndarray:
        return FileNotFoundError
    

    
    
class ActivationFunctions:
    '''
    Class containing activation functions for NN layers.
    '''

    @staticmethod
    def softmax(x :  np.ndarray) -> np.ndarray:
        # Static method for softmax 
        numerator = np.exp(x)
        denominator = np.sum(np.exp(x), axis = 1)
        return numerator / denominator
    

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        # Static method  for relu
        # max(0, x)
        return np.maximum(0, x)




class CostFunctions_derivitives:
    '''
    Class for static methods of cost function derivites for use in neural network.
    '''


    @staticmethod
    def LSE_derivitive(predicted_values: np.ndarray, expected_values : np.ndarray) -> np.ndarray:
        # Calculate LSE cost function between obtained and expected values
        numerator = np.sum(predicted_values - expected_values, axis = 0)
        denominator = predicted_values.shape[0]
        return numerator / denominator


    @staticmethod
    def CEL_function_derivitive(predicted_labels : np.ndarray, actual_labels : np.ndarray) -> np.ndarray:
        return FileNotFoundError
    

    @staticmethod
    def BCEL_function_derivitive(predicted_label : np.ndarray, actual_label : np.ndarray) -> np.ndarray:
        return NotImplementedError
    



class ActivationFunction_Derivities:
    '''CLass for static methods of activation function derivities'''

    @staticmethod
    def relu(x : np.ndarray) -> np.ndarray:
        # Static method for relu deriviative
        return np.where(x > 0, 1, 0)
    

    
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        # Static method for softmax derivitive
        pass