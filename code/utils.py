'''
This file contains auxiliary functions for computing unit vectors and 
rotations.
'''

# import numpy
import numpy as np


def unit_vector(I, D):
    '''
    Compute the Cartesian components of a unit vector 
    as a function of its inclination I and declination D
    
    parameters
    ----------
    I, D: floats - inclination and declination of the unit 
        vector (in degrees)
    
    returns
    -------
    vector: numpy array 1d - unit vector
    '''
    I_rad = np.deg2rad(I)
    D_rad = np.deg2rad(D)
    cosI = np.cos(I_rad)
    sinI = np.sin(I_rad)
    cosD = np.cos(D_rad)
    sinD = np.sin(D_rad)
    vector = np.array([cosI*cosD, cosI*sinD, sinI])
    return vector


def direction(vector):
    """
    Convert a 3-component vector to intensity, inclination and 
    declination.

    parameters
    ----------
    vector : numpy array 1d - the vector.

    returns
    -------
    intensity, inclination, declination: floats - intensity, 
        inclination and declination (in degrees).

    """
    intensity = np.linalg.norm(vector)
    x, y, z = vector
    declination = np.rad2deg(np.arctan2(y, x))
    inclination = np.rad2deg(np.arcsin(z / intensity))
    return intensity, inclination, declination


def rotation_matrix(I, D, dI, dD):
    '''
    Compute the rotation matrix transforming the unit vector
    with inclination I and declination D into the unit vector
    with inclination I + dI and declination D + dD.

    parameters
    ----------
    I, D: floats - inclination and declination (in degrees) of the 
        unit vector to be rotated.
    dI, dD: floats - differences (in degrees) between the 
        inclination and declination of the rotated and original unit 
        vectors.

    returns
    -------
    R: numpy array 2d - rotation matrix.
    '''
    I_rad = np.deg2rad(I)
    D_rad = np.deg2rad(D)

    cosI = np.cos(I_rad)
    sinI = np.sin(I_rad)
    cosD = np.cos(D_rad)
    sinD = np.sin(D_rad)

    dI_rad = np.deg2rad(dI)
    dD_rad = np.deg2rad(dD)

    cosdI = np.cos(dI_rad)
    sindI = np.sin(dI_rad)
    cosdD = np.cos(dD_rad)
    sindD = np.sin(dD_rad)

    I_dI_rad = np.deg2rad(I+dI)
    D_dD_rad = np.deg2rad(D+dD)

    cosI_dI = np.cos(I_dI_rad)
    sinI_dI = np.sin(I_dI_rad)
    cosD_dD = np.cos(D_dD_rad)
    sinD_dD = np.sin(D_dD_rad)

    r00 = sinD_dD*sinD + cosD_dD*cosdI*cosD
    r10 = -cosD_dD*sinD + sinD_dD*cosdI*cosD
    r20 = sindI*cosD
    r01 = -sinD_dD*cosD + cosD_dD*cosdI*sinD
    r11 = cosD_dD*cosD + sinD_dD*cosdI*sinD
    r21 = sindI*sinD
    r02 = -cosD_dD*sindI
    r12 = -sinD_dD*sindI
    r22 = cosdI

    R = np.array([[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]])
    return R