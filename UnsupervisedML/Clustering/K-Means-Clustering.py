import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class K_Means:
    def __init__(self, K : int) -> None:
        # Initalise number of clusters
        self.K = K
    
    def fit(self, X : np.ndarray, max_iters : int = 300) -> np.ndarray:
        # Method for fitting clusters
        # Returns labels of which cluster each point identifies too

        self.X = X
        iter = 0

        self.Cluster_Centers = np.zeros( (self.K ,self.X.shape[1]) )
        labels = np.zeros(self.X.shape[0])
        # Randomly set cluster centers
        for count, _ in enumerate(self.Cluster_Centers):
            self.Cluster_Centers[count, :] = self.X[ np.random.randint( 0 , len(self.X) ) ]
        Not_Converged = True
        # Repeat until Convergence
        while Not_Converged or iter < max_iters:
            iter += 1
            last_centers = np.copy(self.Cluster_Centers)
            for count ,point in enumerate(self.X):
                distances = self._EuclideanDistance(point, self.Cluster_Centers)
                labels[count] = np.argmin(distances)
            
            for count, _ in enumerate(self.Cluster_Centers):
                self.Cluster_Centers[count] = np.mean(self.X[labels == count], axis = 0)


            if np.allclose(last_centers ,self.Cluster_Centers):
                Not_Converged = False
            

        return labels

    @staticmethod
    def _EuclideanDistance( X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Calculate Euclidean Distance between 2 arrays
        return np.linalg.norm(X1 - X2, axis = 1)
    
    
    
 

X, y = make_blobs(n_samples = 150, n_features = 2, random_state = 15, centers = 2)

model = K_Means(2)
sklearn_model = KMeans(2)


labels = model.fit(X)
sklearn_labels = sklearn_model.fit(X)

cluster_one = X[labels == 1]
cluster_two = X[labels == 0]
print(cluster_one)


plt.figure()
plt.scatter(cluster_one[:,0], cluster_one[:, 1])
plt.scatter(cluster_two[:,0], cluster_two[:, 1])
plt.scatter(model.Cluster_Centers[:,0], model.Cluster_Centers[:,1])
plt.scatter(sklearn_model.cluster_centers_[:,0], sklearn_model.cluster_centers_[:,1])
plt.show()