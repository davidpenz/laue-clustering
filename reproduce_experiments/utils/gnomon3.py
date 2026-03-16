import numpy as np


def ComputeGnomon_3(TwiceTheta_Chi, CenterProjection):
    """ compute gnomonic projection coordinates of spot's kf vector defined
    by 2theta chi.
    Adapted from LaueTools.

    From an array:
    [0] array 2theta
    [1] array chi

    returns:
    array:
    [0] gnomonic X
    [1] gnomonic Y
    """
    DEG = np.pi/180.

    data_theta = TwiceTheta_Chi[0] / 2.0
    data_chi = TwiceTheta_Chi[1]

    lat = data_theta*DEG  # in rads
    longit = data_chi*DEG

    centerlat, centerlongit = CenterProjection

    slat0 = np.ones(len(data_chi)) * np.sin(centerlat)
    clat0 = np.ones(len(data_chi)) * np.cos(centerlat)
    longit0 = np.ones(len(data_chi)) * centerlongit

    slat = np.sin(lat)
    clat = np.cos(lat)

    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)

    _gnomonx = clat * np.sin(longit0 - longit) / cosanguldist
    _gnomony = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist

    return _gnomonx, _gnomony
