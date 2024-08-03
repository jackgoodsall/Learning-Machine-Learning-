import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore') 

class LogisticRegression:

    def __init__(self):
        
        self.X : np.ndarray = np.array([])
        self.y : np.ndarray = np.array([])

    def fit(self, X : np.ndarray, y : np.ndarray, alpha : float = 0.005):

        # Uses stochiastic regression to fit logistic regression using a sigmoid funciton
        
        # Create X array
        self.X = np.ones((len(y), X.shape[1] + 1))
        self.X[:, 1:] = X
        # Save y arrays
        self.y = y
        # Create m array
        m : np.ndarray = np.zeros(self.X.shape[1])
        # Stochiastic Gradient descent    
        for count, value  in enumerate(y):
            #print(alpha * (value - self._sigmoid_function(count, m) ) * self.X[count])
            print(self.X[count])
            m = m + alpha * (value - self._sigmoid_function_gradient(count, m) ) * self.X[count]
        return 1/ (1 + np.exp(-np.dot(m, self.X.T)))



    def _sigmoid_function(self, row, m):
        # Calculate value of sigmoid function 
        z = np.dot(m, self.X[row])
        num = 1
        denom = 1 + np.exp( - z )
        return num / denom

    def _sigmoid_function_gradient(self, row, m):
        # Calculate value of the gradient of the sigmoid funciton
        z = np.dot(m, self.X[row])
        num = 1
        denom = 1 + np.exp( - z )
        g_z = num / denom
        gradient = g_z* (1 - g_z)
        return  gradient
    

        


x = np.linspace(-1, 1, 1000).reshape((1000, 1))

y = np.array([1 if i > 0 else 0 for i in x])


model = LogisticRegression()
pred = model.fit(x , y)


plt.figure()
plt.scatter(x , y)
plt.plot(x , pred)
plt.show()