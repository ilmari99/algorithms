import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Ks test for both columns
from scipy.stats import kstest
# Lilliefor composite test
from statsmodels.stats.diagnostic import lilliefors

df = pd.read_csv("t096.csv", header=None, names=['x1', 'x2'])
ALFA_LEVEL = 0.05


df.hist()

# Ks test for both columns
from scipy.stats import kstest
print(df.columns)
x1 = df['x1']
x2 = df['x2']
# The null hypothesis of this KS test is that the data is normally distributed
x1_raw_is_from_std_normal = kstest(x1, 'norm')
x2_raw_is_from_std_normal = kstest(x2, 'norm')

print(f"KS that x1 is from a standard normal distribution: {x1_raw_is_from_std_normal}")
print(f"KS that x2 is from a standard normal distribution: {x2_raw_is_from_std_normal}")

x1_zscore = (x1 - x1.mean()) / x1.std()
x2_zscore = (x2 - x2.mean()) / x2.std()
x1_zscore_is_from_std_normal = kstest(x1_zscore, 'norm')
x2_zscore_is_from_std_normal = kstest(x2_zscore, 'norm')

print(f"KS that x1 standardized is from a standard normal distribution: {x1_zscore_is_from_std_normal}")
print(f"KS that x2 standardized is from a standard normal distribution: {x2_zscore_is_from_std_normal}")


x1_lillie = lilliefors(x1, dist='norm')
x2_lillie = lilliefors(x2, dist='norm')

print(f"LILLE that x1 is from a standard normal distribution: {x1_lillie}")
print(f"LILLE that x2 is from a standard normal distribution: {x2_lillie}")

plt.show()


def distance(x1, x2):
    """Calculates the Euclidean distance between two data points.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

def KL_divergence(p,q):
    """Calculates the Kullback-Leibler divergence between two distributions.

    Args:
        p: A numpy array representing the first distribution.
        q: A numpy array representing the second distribution.

    Returns:
        A float representing the Kullback-Leibler divergence between p and q.
    """
    # remove any values that are 0
    p_is_zero = p == 0
    q_is_zero = q == 0
    combined_is_zero = p_is_zero | q_is_zero
    p = p[~combined_is_zero]
    q = q[~combined_is_zero]
    return np.sum(p * np.log(p / q))

from scipy.special import gamma
class NearestNeighborParameterEstimator:
    """ Take in a dataset, and estimate the PDF of the data using the nearest neighbor method
    assuming that the data is from a normal distribution.
    """
    def __init__(self, data, k=5, distance_function=distance):
        """ Initialize the NearestNeighborParameterEstimator class.
        data: A nxp numpy array, where n is the number of data points and p is the number of dimensions.
        """
        self.data = data
        self.n = data.shape[0]
        self.p = data.shape[1]
        self.k = k
        self.distance = distance_function
    
    def estimate_density(self, x):
        """ Estimate the PDF of a given data point x.
        x: A 1xp numpy array, where p is the number of dimensions.
        """
        # Find the k nearest neighbors
        distances = np.apply_along_axis(self.distance, 1, self.data, x)
        
        sorted_distances = np.sort(distances)
        # reverse
        sorted_distances = sorted_distances[::-1]
        k_nearest_neighbors = sorted_distances[:self.k]
        
        # Calculate the volume of the hypersphere with a radius of distance to the kth nearest neighbor
        volume = np.pi**(self.p / 2) / gamma(self.p / 2 + 1) * k_nearest_neighbors[-1]**self.p
        # Calculate the density
        divider = self.n * volume + 1e-10 # Add a small number to avoid division by zero
        density = self.k / divider
        return density

    def estimate_densities(self, X):
        """ Estimate the PDF of a given data point x.
        X: A nxp numpy array, where n is the number of data points and p is the number of dimensions.
        """
        densities = np.apply_along_axis(self.estimate_density, 1, X)
        return densities
        
    
if __name__ == "__main__":
    data = np.loadtxt("t122.csv", delimiter=",").reshape(-1,1)
    print(data)
    print(f"Basic mean: {data.mean()}")
    print(f"Basic variance: {data.var()}")
    
    normal_distribution_estimated_pdf = np.random.normal(data.mean(), data.var(), data.shape[0]).reshape(-1,1)
    print(f"Normal distribution pdf: {normal_distribution_estimated_pdf}")
    print(f"Sum of normal distribution pdf: {normal_distribution_estimated_pdf.sum()}")
    for K in [1, 3, 5]:
        knn = NearestNeighborParameterEstimator(data, k=K)
        # Calculate the PDF of the data
        densities_at_data_points = knn.estimate_densities(data).reshape(-1,1)
        print(f"KNN estimated densities: {densities_at_data_points}")
        # The densities should sum to 1
        print(f"Sum of densities: {densities_at_data_points.sum()}")
        print(f"Densities mean: {densities_at_data_points.mean()}")
        print(f"Densities variance: {densities_at_data_points.var()}")
        
        # Calculate KL divergence
        kl_divergence = KL_divergence(normal_distribution_estimated_pdf, densities_at_data_points)
        print(f"KL divergence: {kl_divergence}")
        
        
        
        
        