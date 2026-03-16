import numpy as np
from LaueTools.LaueGeometry import unit_q
from LaueTools.findorient import OrientMatrix_from_2hkl
from scipy.spatial.transform import Rotation as R


def OrientMatrix_from_Nhkl(hkls, coords, B, frame="lauetools"):
    """
    Compute crystal orientation matrix from N (>=3) Laue spots.

    Parameters
    ----------
    hkls : list of [h, k, l]
        Miller indices of the reflections.
    coords : list of [2theta, chi]
        Measured coordinates of the reflections (in degrees).
    B : 3x3 numpy array
        Reciprocal lattice matrix (upper triangular, as from CP.calc_B_RR()).
    frame : str
        Coordinate frame for LTGeo.unit_q (default: 'lauetools').

    Returns
    -------
    matorient : 3x3 numpy array
        Best-fit orientation (rotation) matrix.
    """

    # --- 1. Build reciprocal lattice vectors G_i
    G_list = [np.dot(B, np.array(hkl)) for hkl in hkls]
    G_list = [g / np.linalg.norm(g) for g in G_list]  # normalize

    # --- 2. Compute experimental q vectors from detector coordinates
    qq_list = []
    for (twotheta, chi) in coords:
        q = unit_q(twotheta, chi, frame=frame)
        qq_list.append(q / np.linalg.norm(q))  # normalize
    qq_list = np.array(qq_list).T  # shape (3, N)

    # --- 3. Assemble arrays
    G_array = np.array(G_list).T  # shape (3, N)

    # --- 4. Compute best-fit rotation matrix (Kabsch / Procrustes)
    H = qq_list @ G_array.T
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Ensure it's a proper rotation (determinant = +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def angle_difference(rotation1, rotation2):
    difference = rotation1.T @ rotation2
    angle_diff = 180.0 / np.pi * R.magnitude(R.from_matrix(difference))
    return angle_diff


def equivalent_rotation(rotation1, rotation2):
    combined_rotation = rotation1.T @ rotation2
    return np.all(np.isclose(combined_rotation, 0) | np.isclose(np.abs(combined_rotation), 1))


def understand_matorient_convention(hkl1, coord1, hkl2, coord2, B, true_orientation, verbose=0, frame="lauetools"):
    def unit(v):
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)

    def ang_deg(u, v):
        u = unit(u); v = unit(v)
        return np.degrees(np.arccos(np.clip(u @ v, -1, 1)))

    # Build normalized crystal reciprocal directions.
    g1 = unit(B @ np.array(hkl1, float))
    g2 = unit(B @ np.array(hkl2, float))

    # Build normalized lab directions from detector angles.
    q1 = unit(unit_q(coord1[0], coord1[1], frame=frame))
    q2 = unit(unit_q(coord2[0], coord2[1], frame=frame))
    R = OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B, frame=frame)

    # Hypothesis A: R maps crystal -> lab  (q ≈ R g).
    eA = ang_deg(R @ g1, q1) + ang_deg(R @ g2, q2)

    # Hypothesis B: R maps lab -> crystal (g ≈ R q).
    eB = ang_deg(R @ q1, g1) + ang_deg(R @ q2, g2)

    print("Sum error if crystal->lab:", eA)
    print("Sum error if lab->crystal:", eB)
    # The smaller one should be the correct one, and it should be the crystal->lab mapping.

    # Now theck whether the y axis is special.
    def nearest_SO3(M):
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    def axis_angle(R):
        R = nearest_SO3(R)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1)/2, -1, 1)))
        w, V = np.linalg.eig(R)
        axis = np.real(V[:, np.argmin(np.abs(w - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis, ang, R

    Rgt = true_orientation  # your known ground truth orientation matrix

    Delta = R @ Rgt.T
    axis, ang, Delta = axis_angle(Delta)

    print("relative angle (deg):", ang)
    print("axis (lab):", axis)

    # check if it's y
    print("abs(axis·y):", abs(axis @ np.array([0.,1.,0.])))

    # check directly against a y-180 rotation matrix
    Ry = np.diag([-1, 1, -1])
    print("||Delta - Ry||:", np.linalg.norm(Delta - Ry))

    print("Delta (rounded):\n", np.round(Delta, 3))

    Ry = np.diag([-1, 1, -1])  # 180° rotation about y
    Fy = np.diag([ 1,-1,  1])  # flip y axis (reflection)

    candidates = {
        "none": R,
        "Ry @ R": Ry @ R,
        "Fy @ R": Fy @ R,
        "R @ Ry": R @ Ry,
        "R @ Fy": R @ Fy,
    }

    for name, Rc in candidates.items():
        err = np.linalg.norm(Rc - Rgt)
        det = np.linalg.det(Rc)
        print(f"{name:6s}  ||Rc-Rgt||={err:.6g}   det={det:.6g}")
