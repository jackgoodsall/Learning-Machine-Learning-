import numpy as np
import matplotlib.pyplot as plt
import torch

class LinearRegression:

    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])

    def fit(self, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        # Initalise a X array of size (n + 1, m) and (1,m) is a column of 1s

        self.X = np.ones(( len(y) , X.shape[1] + 1 ,))
        self.X[:, 1:] = X
        print(self.X)
        self.y = y

        # Calculate w = (X.T X) ^-1 X.T y

        w = np.dot( np.linalg.pinv(np.dot(self.X.T, self.X)) , np.dot(self.X.T, self.y))
        print(w.shape, self.X.shape)
        return np.dot(self.X, w)
    
X = np.linspace((0, 10, 50), (100, 200, 1000), 200)
y = np.linspace(0, 1000, 200) 



model = LinearRegression()
pred = model.fit(X, y)

plt.figure()
plt.scatter(X[:, 0], y)
plt.plot(X[:,0 ], pred)
plt.show()


