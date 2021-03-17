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


# NAÏVE APPROACH
def naive(P, S):
    assert P.ndim == S.ndim == 2, 'P and S must be 2d arrays'
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


# AVOIDING SQUARE ROOTS
def avoid_sqrt(P, S):
    assert P.ndim == S.ndim == 2, 'P and S must be 2d arrays'
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


# AVOIDING FOR LOOPS
def avoid_sqrt_inner_loops(P, S):
    assert P.ndim == S.ndim == 2, 'P and S must be 2d arrays'
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'
    # compute components of matrix D
    D1 = np.sum(a=P*P, axis=0)
    D2 = np.sum(a=S*S, axis=0)
    D3 = 2*np.dot(P.T, S)

    D = D1[:,np.newaxis] + D2[np.newaxis,:] - D3

    return D


# OPTIMIZED SCIPY FUNCTION
def scipy_distance(P, S):
    assert P.ndim == S.ndim == 2, 'P and S must be 2d arrays'
    V = spt.distance.cdist(P.T, S.T, 'sqeuclidean')
    #return spt.distance.squareform(V, force='tomatrix')
    return V


# NAÏVE APPROACH WITH NUMBA
@njit
def naive_numba(P, S):
    assert P.ndim == S.ndim == 2, 'P and S must be 2d arrays'
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


# AVOIDING SQUARE ROOTS WITH NUMBA
@njit
def avoid_sqrt_numba(P, S):
    assert P.ndim == S.ndim == 2, 'P and S must be 2d arrays'
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