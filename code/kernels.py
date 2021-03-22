'''
This file contains Python codes for computing the gravitational field of
point masses and the magnetic induction field of dipoles.
'''

# import numpy
import numpy as np
# import sedm
import sedm


def inverse_distance(P, S):
    '''
    Compute the inverse distance between the observation points 
    and the sources.

    parameters
    ----------
    P: numpy array 2d - 3 x N matrix containing the coordinates 
        x (1rt row), y (2nd row), z (3rd row) of N observation points.
        The ith column contains the coordinates of the ith observation point.
    S: numpy array 2d - 3 x M matrix containing the coordinates 
        x (1rt row), y (2nd row), z (3rd row) of M sources.
        The jth column contains the coordinates of the jth source.


    returns
    -------
    K: numpyy array 2d - N x M matrix containing the inverse distance values.
    '''

    R2 = sedm.avoid_sqrt_inner_loops(P, S)

    K = 1/np.sqrt(R2)
    
    return K


def first_derivatives(P, S):
    '''
    Compute the first derivatives of the inverse
    distance function between the observation points 
    and the sources.

    parameters
    ----------
    P: numpy array 2d - 3 x N matrix containing the coordinates 
        x (1rt row), y (2nd row), z (3rd row) of N observation points.
        The ith column contains the coordinates of the ith observation point.
    S: numpy array 2d - 3 x M matrix containing the coordinates 
        x (1rt row), y (2nd row), z (3rd row) of M sources.
        The jth column contains the coordinates of the jth source.


    returns
    -------
    Kx, Ky, Kz: numpyy arrays 2d - N x M matrices 
    containing the first derivatives x, y, z.
    '''

    R2 = sedm.avoid_sqrt_inner_loops(P, S)
    R3 = R2*np.sqrt(R2)

    X = P[0][:, np.newaxis] - S[0]
    Y = P[1][:, np.newaxis] - S[1]
    Z = P[2][:, np.newaxis] - S[2]

    Kx = -X/R3
    Ky = -Y/R3
    Kz = -Z/R3
    
    return Kx, Ky, Kz


def second_derivatives(P, S):
    '''
    Compute the second derivatives of the inverse
    distance function between the observation points 
    and the sources.

    parameters
    ----------
    P: numpy array 2d - 3 x N matrix containing the coordinates 
        x (1rt row), y (2nd row), z (3rd row) of N observation points.
        The ith column contains the coordinates of the ith observation point.
    S: numpy array 2d - 3 x M matrix containing the coordinates 
        x (1rt row), y (2nd row), z (3rd row) of M sources.
        The jth column contains the coordinates of the jth source.


    returns
    -------
    Kxx, Kxy, Kxz, Kyy, Kyz: numpyy arrays 2d - N x M matrices 
    containing the second derivatives xx, xy, xz, yy and yz 
    (zz is not computed because it can be obtained from xx and yy).
    '''

    R2 = sedm.avoid_sqrt_inner_loops(P, S)
    R3 = R2*np.sqrt(R2)
    R5 = R3*R2

    X = P[0][:, np.newaxis] - S[0]
    Y = P[1][:, np.newaxis] - S[1]
    Z = P[2][:, np.newaxis] - S[2]

    Kxx = (3*X*X)/R5 - 1/R3
    Kxy = (3*X*Y)/R5
    Kxz = (3*X*Z)/R5
    Kyy = (3*Y*Y)/R5 - 1/R3
    Kyz = (3*Y*Z)/R5
    
    return Kxx, Kxy, Kxz, Kyy, Kyz


def matrices_A(F, Kxx, Kxy, Kxz, Kyy, Kyz):
    '''
    Compute the matrices Ax, Ay and Az containing,
    respectively, the x, y and z components of the
    vector defined by the product of the main-field 
    unit vector and the gradient tensor of the inverse
    distance function.

    parameters
    ----------
    F: numpy array 1d - unit vector defining the main-field
        direction.
    Kxx, Kxy, Kxz, Kyy, Kyz: numpyy arrays 2d - N x M matrices 
        containing the second derivatives xx, xy, xz, yy and yz 
        (zz is not computed because it can be obtained from xx 
        and yy).

    returns
    -------
    Ax, Ay, Az: numpy array 2d - matrices containing the 
        x, y and z components of the unit vector F and 
        2nd derivatives matrix of the inverse distance function.
    '''
    Ax = F[0]*Kxx + F[1]*Kxy + F[2]*Kxz
    Ay = F[0]*Kxy + F[1]*Kyy + F[2]*Kyz
    Az = F[0]*Kxz + F[1]*Kyz - F[2]*(Kxx + Kyy)

    return Ax, Ay, Az


def magnitude_A(Ax, Ay, Az):
    '''
    Compute the matrix whose element ij is the magnitude of the vector
    defined by the product of the main-field unit vector and the gradient 
    tensor of the inverse distance function.

    parameters
    ----------
    Ax, Ay, Az: numpy array 2d - matrices containing the 
        x, y and z components of the.

    returns
    -------
    M: numpyy array 2d - matrix of magnitudes.
    '''
    M = np.sqrt(Ax*Ax + Ay*Ay + Az*Az)

    return M


def dipole_matrix(h, Ax, Ay, Az):
    '''
    Compute the matrix whose element ij is the approximated 
    total-field anomaly produced by a dipole with total 
    magnetization vector h.

    parameters
    ----------
    h: numpy array 1d - unit vector defining the 
        total magnetization of the dipoles.

    Ax, Ay, Az: numpy arrays 2d - matrices depending on the 
        direction of the main field and the second derivatives
        of the inverse distance function.

    returns
    -------
    G: numpy array 2d - dipole matrix.
    '''
    G = h[0]*Ax + h[1]*Ay + h[2]*Az

    return G



