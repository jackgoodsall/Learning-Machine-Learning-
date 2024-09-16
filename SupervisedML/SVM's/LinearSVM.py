import numpy as np

class SupportVectorMachine:
    '''
    Class for a linear support vector machine
    '''

    def __init__(self) -> None:
        pass

    
    def train(self, X_data : np.ndarray , y_data : np.ndarray) -> None:
        '''
        Training method for the SVM class.
        Arguments:
        X_data : np.ndarray (n_training_examples, n_features)
        y_data : np.ndarray (n_training_examples, n) 
        '''
        