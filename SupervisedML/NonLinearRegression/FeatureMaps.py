import numpy as np
import matplotlib.pyplot as plt

class NonLinearRegression:

    def __init__(self, featureMap) -> None:
        # initalise a non linear regression model with a feature map function
        self.featuremap   = featureMap

    def fit(self, x : np.ndarray, y : np.ndarray, learnrate: float = 0.01, epoches : int  = 1000) -> None:
        
        # Initalise 
        # number of samples
        n : int = x.shape[0]
        # number of features
        m : int = x.shape[1]
        # Initalise training array
        x_train : np.ndarray = np.ones((n, m + 1))
        y_train : np.ndarray = y.reshape((n, 1))

        x_train[:,1:] = x
        # Batch gradient descent
        theta : np.ndarray= np.zeros((m + 1, 1))
        for _ in range(epoches):
            mapped_values = self.featuremap(x_train)
            theta = theta + learnrate * np.sum( (y_train - mapped_values) * mapped_values)
        
        self.theta = theta

    def predict(self, x : np.ndarray) -> np.ndarray:
        # number of samples
        n : int = x.shape[0]
        # number of features
        m : int = x.shape[1]
        # Initalise training array
        x_predit : np.ndarray = np.ones((n, m + 1))
        x_predit[:, 1:] = x


        return np.dot(self.featuremap(x_predit), self.theta) 



def cubic_feature_map(x : np.ndarray) -> float | np.ndarray:

    return x**3 + x**2 + x

x : np.ndarray = np.linspace(0, 100, 1000).reshape((1000, 1))
y : np.ndarray = x**3 + 2 * x**2 



model = NonLinearRegression(cubic_feature_map)
model.fit(x, y)

pred = model.predict(x)

plt.figure()
plt.scatter(x, y)
plt.plot(x, pred)
plt.show()