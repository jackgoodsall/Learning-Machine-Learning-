import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore') 

class LogisticRegression:

    def __init__(self) -> None:
        
        pass

    def fit(self, X : np.ndarray, y : np.ndarray, alpha : float = 1, epoches : int = 100) -> None:

        # Uses stochiastic regression to fit logistic regression using a sigmoid funciton
        
        # m is the number of 
        m : int = X.shape[0]
        # n number of features
        n : int = X.shape[1]

        # Create X array shape (m, n + 1)
        X_train : np.ndarray = np.ones((m, n  + 1))
        
        # X_0 = 1
        X_train[:, 1:] = X

        # Save y values and ensure it is a matrix (could be an array other wise and seems to break iot )
        y_train : np.ndarray = y.reshape(m, 1)
        # Create theta array
        theta : np.ndarray = np.zeros((n + 1, 1))
        
        # Stochiastic Gradient descent    
        for _ in range(epoches):    
            for count, value  in enumerate(y):
                row = X_train[count, ].reshape((1, n + 1))
                yhat = float(self._sigmoid_function(row, theta))
                theta = theta + alpha * (value - yhat ) * (yhat) * ( 1- yhat) * row.T
            
        self.theta : np.ndarray = theta
        return None
        



    def _sigmoid_function(self, row, m) -> float | np.ndarray:
        # Calculate value of sigmoid function 
        z = np.dot(row, m)
        num = 1
        denom = 1 + np.exp( - z )
        return num / denom

    def predict(self, X):
        # Model to predict outcome after the model has been taught
         # m is the number of 
        m : int = X.shape[0]
        # n number of features
        n : int = X.shape[1]

        # Create X array shape (m, n + 1)
        X_test : np.ndarray = np.ones((m, n  + 1))
        X_test[:, 1:] = X


        return self._sigmoid_function(X_test, self.theta)
        


x = np.linspace(-1, 1, 1000).reshape((1000, 1))

y = np.array([1 if i > 0 else 0 for i in x])


model = LogisticRegression()
model.fit(x , y)
predictions = model.predict(x)


plt.figure()
plt.scatter(x , y)
plt.plot(x , predictions)
plt.show()