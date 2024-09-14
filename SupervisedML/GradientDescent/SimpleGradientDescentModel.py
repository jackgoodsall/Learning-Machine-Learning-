"""
First attempt at a gradient descent, for now will assume only 2 paramaters
Not really looked at any resources for this yet so just haivng fun with it trying to
solve a problem without enough information, will definately not be efficent.
"""

import numpy as np
import matplotlib.pyplot as plt


class GradientDescentModel:

    def __init__(self):
        # Making it same style as models from sklearn and tensor flow, so first create the model
        # then use fit method to fit to data
        self.X = np.array([], dtype = float)
        self.y = np.array([], dtype = float)
       
    def fit(self, X, y ,learn_rate = 0.0001, epoches = 10000000):
        # learn_rate and epoches are hyper parmaters so will just give them default parameters
        # assume 1D x and 1D y for now and for now assume a linear function ( way easier to play around with)

        self.X = X
        self.y = y
        w = 0
        b = 0

        for _ in range(epoches):
            dw =  self._cost_function_w_derivate(w, b)
            db =  self._cost_function_b_derivate(w, b)
            w = w - learn_rate * dw
            b = b - learn_rate  * db

        return self._linear_function(w, b)
    
    def _linear_function(self, w, b):
        # Linear model
        return w * self.X + b

    def _cost_function(self, w, b):
        # cost function
        cost_sumation = np.sum((self._linear_function(w, b)  - self.y)**2)
        return cost_sumation / (2* np.len(self.y))
    
    def _cost_function_b_derivate(self, w, b):
        derivate_sumation = np.sum(self._linear_function(w, b) - self.y)
        return derivate_sumation / len(self.y)
    
    def _cost_function_w_derivate(self, w, b):
        derivate_sumation = np.sum((self._linear_function(w, b) - self.y) * self.X)
        return derivate_sumation / len(self.y)
        
    

x = np.linspace(0, 100, 1000)
y = np.linspace(30, 5000, 1000) + np.random.gamma(1000, 20)

model = GradientDescentModel()
pred =  model.fit(x, y)

plt.figure()
plt.plot(x, pred)
plt.scatter(x, y)
plt.show()
