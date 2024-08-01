import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self):

        # Initalise class

        self.X = np.array([])
        self.y = np.array([])
        
    def fit(self, X, y):
        # Fit data to a linear model of 2 parameters
        self.X = X
        self.y = y      
        # Calculate values of m and b
        m = self.calculate_m()
        b = self.calculate_b(m)
        return (m, b)

    def calculate_b(self, m):
        # Calculate optimal value of b based on mean squared error
        b  = np.mean(self.y) - m * np.mean(X)
        return b
    
    def calculate_m(self):
        # Calculate optimal value of m based on mean squared error
        n = len(y)
        # Calulate numerator of the equation
        num = n * np.sum(self.X * self.y) - np.sum(self.X) * np.sum(self.y)
        # Calculate denominator
        denom = n * np.sum(self.X **2) -(np.sum(self.X))**2
        return num/denom if denom != 0 else 0

X = np.linspace(0, 100, 200)
y = np.linspace(0, 1000, 200) + np.random.exponential(100, size = 200)


oneparameter_model = LinearRegression()
pars = oneparameter_model.fit(X, y)


plt.figure()
plt.plot(X, y)
plt.plot(X, pars[0] * X + pars[1])
plt.show()