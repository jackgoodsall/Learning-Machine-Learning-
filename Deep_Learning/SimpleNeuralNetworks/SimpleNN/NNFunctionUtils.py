import numpy as np

class FunctionUtils:
    '''
        Class contain static methods need for NN architecture
    '''

    @staticmethod
    def cost_function(obtained_values: np.ndarray, expected_values : np.ndarray) -> np.ndarray:
        # Calculate LSE cost function between obtained and expected values

        numerator = np.sum((obtained_values - expected_values)**2, axis = 0)
        denom  =  2 
        return numerator/ denom
    

    @staticmethod
    def cost_function_derivitive(obtained_values: np.ndarray, expected_values : np.ndarray) -> np.ndarray:
        # Calculate LSE cost function between obtained and expected values

        numerator = np.sum(obtained_values - expected_values, axis = 0)
        return numerator
    

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        # Static method  for relu
        # max(0, x)
        return np.maximum(0, x)
    
    
    @staticmethod
    def relu_derivitive(x : np.ndarray) -> np.ndarray:
        # Static method for relu deriviative
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x :  np.ndarray) -> np.ndarray:
        # Static method for softmax 
        pass
    

    