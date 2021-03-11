'''
This file contains Python codes for computing the Squared Euclidean Distance
Matrix between points based on the functions proposed by Bauckhage (2014).

Bauckhage, C. (2014). NumPy / SciPy Recipes for Data Science: Squared Euclidean
Distance Matrices, https://dx.doi.org/10.13140/2.1.4426.1127
'''

# import numpy
import numpy as np
# import numpy linear algebra module
import numpy.linalg as la
# import scipy spatial module
import scipy.spatial as spt
# import numba
from numba import njit

# METHOD 1: A NAÏVE APPROACH
def naive(P, S):
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'
    # initialize squared EDM D
    D = np.zeros((Np,Ns))
    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            D[i,j] = la.norm(P[:,i] - S[:,j])**2
    return D


# METHOD 2: AVOIDING SQUARE ROOTS
def avoid_sqrt(P, S):
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'
    # initialize squared EDM D
    D = np.zeros((Np,Ns))
    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            d = P[:,i] - S[:,j]
            D[i,j] = np.dot(d, d)
    return D



# METHOD 3: AVOIDING REPEATED INNER PRODUCTS
def avoid_sqrt_inner(P, S):
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'
    # compute Gram matrix
    G = np.dot(P.T, S)
    # initialize squared Euclidean distance matrix
    D = np.zeros((Np,Ns))
    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            # make use of |a-b|ˆ2 = a’a + b’b - 2a’b
            D[i,j] = G[i,i] - 2*G[i,j] + G[j,j]
    return D


# METHOD 4: AVOIDING FOR LOOPS
def avoid_sqrt_inner_loops(P, S):
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'
    # compute Gram matrix
    G = np.dot(P.T, S)
    # compute matrix H
    H = np.tile(np.diag(G), (n,1))
    return H + H.T - 2*G


# METHOD 5: RESORTING TO BUILD-IN FUNCTIONS
def scipy_distance(P, S):
    V = spt.distance.cdist(P.T, S.T, 'sqeuclidean')
    return spt.distance.squareform(V)


# METHOD 6: USING NUMBA
@njit
def avoid_sqrt_inner_jit(P, S):
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    # compute Gram matrix
    G = np.dot(P.T, S)
    # initialize squared Euclidean distance matrix
    D = np.zeros((Np,Ns))
    # only iterate over upper trianlge
    for i in range(Np):
        for j in range(i+1,Ns):
            # make use of |a-b|ˆ2 = a’a + b’b - 2a’b
            D[i,j] = G[i,i] - 2*G[i,j] + G[j,j]
            D[j,i] = D[i,j]
    return D
