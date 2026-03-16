import numpy as np
from LaueTools.generaltools import calculdist_from_thetachi


def compute_codebars(s_tth, s_chi, bins, normalize=False):
    """Compute the angle distance frequencies/distribution for the given laue data.

    Parameters
    ----------
    s_tth : array-like of shape(n_points)
        Array of 2theta values.
    s_chi : array-like of shape(n_points)
        Array of chi values
    bins : array-like of shape(n_bins)
        The bins for the frequencies.
    normalize : bool
        If set to True, the resulting vectors are normalized.

    Returns
    -------
    codebars : np.ndarray of shape (n_points, n_bins)
    """
    s_theta = np.array(s_tth) / 2
    all_points = np.array(list(zip(s_theta, s_chi)))
    all_pairs_angle_dist = calculdist_from_thetachi(all_points, all_points)
    codebars = []
    for i in range(len(all_pairs_angle_dist)):
        angles = all_pairs_angle_dist[i]
        fingerprint = np.histogram(angles, bins=bins)[0]
        if normalize:
            fingerprint = fingerprint / np.max(fingerprint)
        codebars.append(fingerprint)
    codebars = np.array(codebars)

    return codebars
