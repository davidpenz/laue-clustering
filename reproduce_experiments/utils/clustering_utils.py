import itertools
import os
from collections import defaultdict
from pathlib import Path

import LaueTools.CrystalParameters as CP
import LaueTools.dict_LaueTools as dictLT
import LaueTools.lauecore as LT
import scipy
import torch
from LaueTools.findorient import OrientMatrix_from_2hkl
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from reproduce_experiments.datasets import EvalDataset
from reproduce_experiments.utils.gnomon3 import ComputeGnomon_3
from reproduce_experiments.utils.features import compute_codebars
from reproduce_experiments.utils.heuristics import *
from reproduce_experiments.utils.physics_utils import understand_matorient_convention
from reproduce_experiments.utils.general_utils import setup_model


CLUSTER_HEURISTICS = {
    'mean_confidence': cluster_heuristic_mean_confidence,
    'top-k': cluster_heuristic_top_k_confidence,
}


def round_point(pt, decimals=3):
    return tuple(np.round(pt, decimals))


def lines_approx_equal(lines1, lines2, threshold=0.1):
    if not lines1 or not lines2:
        return False
    intersection = len(lines1 & lines2)
    min_len = min(len(lines1), len(lines2))
    return intersection / min_len >= threshold


def getAngle(P, Q):
    R = P @ Q.T
    cos_theta = (np.trace(R) - 1) / 2
    if np.isclose(cos_theta, -1):
        cos_theta = -1
    if np.isclose(cos_theta, 1):
        cos_theta = 1
    return np.arccos(cos_theta) * (180 / np.pi)


def load_test_data(data_config: dict, data_directory=None, give_us_all=True, shuffle=False, give_us_some=True):
    material = data_config.get("material_", [None])[0]
    n_rotations = data_config.get("nb_grains_per_lp_mat", [None])[0]

    print(f"Loading {material} dataset with {n_rotations} grains.")

    # Load general data
    if data_directory is None:
        data_dir = Path(os.getcwd()) / 'data' / f'{material}_{n_rotations}grain'
    else:
        data_dir = data_directory / f'{material}_{n_rotations}grain'
    classhkl = np.load(data_dir / 'MOD_grain_classhkl_angbin.npz')['arr_0']
    ang_bins = np.load(data_dir / 'MOD_grain_classhkl_angbin.npz')['arr_1']
    class_filter = np.load(data_dir / 'MOD_grain_classhkl_angbin.npz')['arr_2']

    # Load Laue data
    test_data_dir = data_dir / 'testing_data_new'

    dataset = EvalDataset(test_data_dir, give_us_all=give_us_all, give_us_some=give_us_some)
    # TODO: change back the shuffling to False.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return dataloader, classhkl, ang_bins, class_filter, material, n_rotations


def load_model(data_config: dict, model_config: dict, ang_bins_length: int, classhkl_length: int, device=torch.device("cuda:0")):
    model_class = model_config["model_class"]
    input_dropout = model_config["input_dropout"]
    dims = model_config["dims"]
    activation = model_config["activation"]

    material = model_config.get("material", None)
    n_rotations = model_config.get("n_rotations", None)

    if model_config["material"] is None:
        material = data_config.get("material_", [None])[0]
    if model_config["n_rotations"] is None:
        n_rotations = data_config.get("nb_grains_per_lp_mat", [None])[0]

    print(f"Loading {material} model with {n_rotations} grain(s).")

    # Load and initialize model
    model_path = Path(os.getcwd()) / 'trained_models' / f'{material}_{n_rotations}grain_CustomNN'
    model = setup_model(model_class, device, ang_bins_length - 1, classhkl_length, input_dropout, dims, activation,
                        verbose=False)
    if device == torch.device("cpu"):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def compute_gnomonic_projection(s_tth, s_chi):
    # Compute projection and visualize.
    DEG = np.pi / 180.
    # Coordinates of the center of the projection in degrees, need to pass 2theta/2.
    center_points = (110 / 2 * DEG, -37 * DEG)
    X_gno, Y_gno = ComputeGnomon_3((s_tth, s_chi), CenterProjection=center_points)
    # Contains projected coordinates.
    points = np.vstack((X_gno, Y_gno)).T

    return points


def compute_hough_embeddings(points, s_grain_id, n_bins, r=0.3, eps=1e-12):
    """
    Computes Hough line embeddings for nearby point pairs using a KD-tree to avoid
    constructing the full O(N^2) pairwise distance matrix.
    For each local pair, the line angle and signed normal distance are computed
    and accumulated into a 2D histogram in (angle, distance) space.
    """
    points = np.asarray(points, dtype=np.float64)
    s_grain_id = np.asarray(s_grain_id)

    # Consider close pirs only.
    tree = cKDTree(points)
    pairs = np.array(list(tree.query_pairs(r=r)), dtype=np.int32)  # Shape is (E, 2).
    if pairs.size == 0:
        # Return empty but well-formed outputs
        h = np.zeros((n_bins, n_bins), dtype=np.int64)
        return (np.empty((0, 2)), [], {}, h,
                np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    i = pairs[:, 0]
    j = pairs[:, 1]

    # Vectorized deltas.
    delta = points[j] - points[i]                      # (E, 2)
    dx = delta[:, 0]
    dy = delta[:, 1]

    # Skip (almost) identical points.
    good = (dx*dx + dy*dy) > eps
    i, j = i[good], j[good]

    # Compute embedding
    u = points[:,0]
    v = points[:,1]
    angle = np.arctan2(u[j] - u[i], v[i] - v[j])
    magnitude = v[i] * np.sin(angle) + u[i] * np.cos(angle)

    # Shift angle to get old results
    angle = angle + np.pi/2

    # Canonize embeddings
    # TODO: Fix fix
    angle = np.mod(angle, np.pi)
    normal = np.abs(magnitude)

    edge_embeddings = np.stack([angle, normal], axis=1)

    same = (s_grain_id[i] == s_grain_id[j])
    colors = np.where(same, s_grain_id[i], -1).tolist()

    edge_to_points = {k: [int(a), int(b)] for k, (a, b) in enumerate(zip(i, j))}

    h, angle_edges, dist_edges = np.histogram2d(
        edge_embeddings[:, 0], edge_embeddings[:, 1], bins=n_bins
    )

    return edge_embeddings, colors, edge_to_points, h, angle_edges, dist_edges


def compute_lines(n_lines, h, angle_values, distance_values, edge_to_points, points, edge_embeddings):
    """
    Faster version of the compute_lines function.
    compute_lines does: for every chosen line bin, let's scan every edge and check if it's inside the bin.
    Here we assign every edge to a bin (once), then have an index from bins -> list of edges, so to
    compute_lines_fast does: let us assign each edge to its corresponding Hough histogram bin (once!)
    and build a lookup from bins to the edges they contain.
    When selecting the top n_lines, retrieve all edges belonging to the chosen bin directly from this index
    instead of scanning all edges.
    Result should be identical to compute_lines, just faster.
    TODO: remove points variable.
    """
    # Convert edge_to_points dict -> endpoint arrays aligned to edge_embeddings rows ---
    # We use edge_key = len(edge_embeddings) so keys should be (0, ..., E-1).

    # If empty embeddings.
    E = edge_embeddings.shape[0]
    if E == 0:
        return [], defaultdict(set)

    keys = list(edge_to_points.keys())
    if len(keys) != E:
        raise ValueError(f"edge_to_points has {len(keys)} entries but edge_embeddings has {E} rows.")
    keys_sorted = np.array(sorted(keys), dtype=np.int64)

    edge_i = np.empty(E, dtype=np.int32)
    edge_j = np.empty(E, dtype=np.int32)
    for out_k, k in enumerate(keys_sorted):
        a, b = edge_to_points[int(k)]
        edge_i[out_k] = a
        edge_j[out_k] = b

    # Reorder embeddings to match the sorted keys (usually keys already 0..E-1 so this is a no-op).
    emb = edge_embeddings[keys_sorted]

    # Pre-bin all edges ONCE.
    a = emb[:, 0]
    d = emb[:, 1]
    nA = len(angle_values) - 1
    nD = len(distance_values) - 1

    ai = np.searchsorted(angle_values, a, side="right") - 1
    di = np.searchsorted(distance_values, d, side="right") - 1

    valid = (ai >= 0) & (ai < nA) & (di >= 0) & (di < nD)
    ai = ai[valid]
    di = di[valid]

    # Keep only valid edges for lookup.
    edge_i_v = edge_i[valid]
    edge_j_v = edge_j[valid]
    emb_v = emb[valid]

    # Flat bin id for each valid edge.
    flat_bin = ai * nD + di
    n_bins_total = nA * nD

    # Build bin -> edges index via sorting + prefix sums.
    order = np.argsort(flat_bin, kind="mergesort")
    flat_sorted = flat_bin[order]
    counts = np.bincount(flat_sorted, minlength=n_bins_total)
    starts = np.empty(n_bins_total + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])

    # And round stuff.
    a_round = np.round(emb_v[:, 0], 2)
    d_round = np.round(emb_v[:, 1], 2)

    all_explained_idx = []
    point_to_lines_idx = defaultdict(set)

    # Main loop: pick best bin from h, then fetch its edges .
    for _ in range(n_lines):
        flat_h = int(np.argmax(h))
        if h.flat[flat_h] == 0:
            break
        x, y = divmod(flat_h, h.shape[1])

        # Map (x,y) to flat bin id used by our edge binning.
        b = x * nD + y

        # And retrieve edges in that bin.
        s, t = int(starts[b]), int(starts[b + 1])
        if s != t:
            eids = order[s:t]  # Indices into emb_v/edge_i_v/edge_j_v.

            # All endpoints explained by this "line".
            pts = np.concatenate([edge_i_v[eids], edge_j_v[eids]])
            unique_pts = np.unique(pts)
            all_explained_idx.extend(unique_pts.tolist())

            # Record params per point.
            for e in eids:
                param = (float(a_round[e]), float(d_round[e]))
                point_to_lines_idx[int(edge_i_v[e])].add(param)
                point_to_lines_idx[int(edge_j_v[e])].add(param)

        # Zero out selected bin.
        h[x, y] = 0

    return all_explained_idx, point_to_lines_idx


def compute_star_centers(star_center_count, n_lines, h, angle_values, distance_values, edge_to_points, points, edge_embeddings):
    # Find star centers (points explained by >2 lines). But fast.
    all_explained_idx, point_to_lines_idx = compute_lines(
        n_lines, h, angle_values, distance_values, edge_to_points, points, edge_embeddings
    )
    if len(all_explained_idx) == 0:
        return np.array([], dtype=np.int32), point_to_lines_idx

    unique_explained_idx, counts = np.unique(all_explained_idx, return_counts=True)
    star_center_idx = unique_explained_idx[counts >= star_center_count]
    return star_center_idx, point_to_lines_idx


def compute_cluster_candidates(star_center_idx, point_to_lines_idx, cid):
    # Build initial clusters from star centers.
    clusters = []

    # Check if enough star centers are found
    if len(star_center_idx) < 1:
        print("No star centers found at iteration " + str(cid+1))
        return clusters

    for center in star_center_idx:
        center_lines = point_to_lines_idx.get(center, set())
        cluster = [pt for pt, lines in point_to_lines_idx.items() if lines_approx_equal(lines, center_lines)]
        if cluster:
            clusters.append(set(cluster))

    # Sort clusters by size
    # TODO: sort by another metric?
    clusters = sorted(clusters, key=len, reverse=True)

    return clusters


def get_cluster(clusters, laue_spots, ang_bins, classhkl, location, heuristic, model, device, bad_cluster_idxs=None):
    all_confidences = []
    correct_idxs = []
    all_hkls = []

    all_y_true, all_y_pred = None, None

    #if bad_cluster_idxs is not None:
    #    if len(bad_cluster_idxs) > 0:
    #        cidx_to_keep = list(range(len(clusters)))
    #        cidx_to_keep = [c for c in cidx_to_keep if c not in bad_cluster_idxs]
    #        clusters = list(np.array(clusters)[cidx_to_keep])

    for i, cluster in enumerate(clusters):
        cluster_points_idx = np.array(list(cluster))
        cluster_points = laue_spots[cluster_points_idx]

        # Compute feature vectors only based on one cluster
        X = compute_codebars(cluster_points[:, 0], cluster_points[:, 1], ang_bins, True)

        # Predict labels and compute accuracy
        y_pred = model(torch.tensor(X, dtype=torch.float).to(device)).detach().cpu().numpy()
        y_true = location[cluster_points_idx]

        if all_y_pred is None:
            all_y_pred = y_pred.argmax(axis=1)
        else:
            all_y_pred = np.concatenate((all_y_pred, y_pred.argmax(axis=1)), axis=0)

        if all_y_true is None:
            all_y_true = y_true
        else:
            all_y_true = np.concatenate((all_y_true, y_true), axis=0)

        softmax_outputs = scipy.special.softmax(y_pred, axis=1)
        confidences = np.max(softmax_outputs, axis=1)
        correct_idx = y_pred.argmax(axis=1) == y_true
        correct_idxs.append(correct_idx)
        all_confidences.append(confidences)
        all_hkl = classhkl[y_pred.argmax(axis=1)]
        all_hkls.append(all_hkl)

    best_cluster_idx = CLUSTER_HEURISTICS[heuristic](all_confidences)


    best_cluster = clusters[best_cluster_idx]
    best_all_confidences = all_confidences[best_cluster_idx]
    best_correct_idx = correct_idxs[best_cluster_idx]
    best_hkl = all_hkls[best_cluster_idx]

    best_cluster_points_idx = np.array(list(best_cluster))
    best_cluster_points = laue_spots[best_cluster_points_idx]

    return best_cluster, best_cluster_points_idx, best_cluster_points, best_all_confidences, best_correct_idx, best_hkl, all_y_true, all_y_pred, best_cluster_idx


def get_cluster_new(
        clusters, laue_spots, ang_bins, classhkl, location, heuristic, model, device,
        material, cid, point_heuristic_dict, s_grain_id, max_distance_threshold,
        always_include_center=False, data_config=None
):
    all_confidences = []
    correct_idxs = []
    all_hkls = []

    all_y_true, all_y_pred = None, None

    all_number_explained_points = []

    for i, cluster in enumerate(clusters):
        cluster_points_idx = np.array(list(cluster))
        cluster_points = laue_spots[cluster_points_idx]

        X = compute_codebars(cluster_points[:, 0], cluster_points[:, 1], ang_bins, True)

        y_pred = model(torch.tensor(X, dtype=torch.float).to(device)).detach().cpu().numpy()
        y_true = location[cluster_points_idx]

        if all_y_pred is None:
            all_y_pred = y_pred.argmax(axis=1)
        else:
            all_y_pred = np.concatenate((all_y_pred, y_pred.argmax(axis=1)), axis=0)

        if all_y_true is None:
            all_y_true = y_true
        else:
            all_y_true = np.concatenate((all_y_true, y_true), axis=0)

        softmax_outputs = scipy.special.softmax(y_pred, axis=1)
        confidences = np.max(softmax_outputs, axis=1)
        correct_idx = y_pred.argmax(axis=1) == y_true
        correct_idxs.append(correct_idx)
        all_confidences.append(confidences)
        all_hkl = classhkl[y_pred.argmax(axis=1)]
        all_hkls.append(all_hkl)

        cluster_points_idx = np.array(list(cluster))
        cluster_points = laue_spots[cluster_points_idx]

        try:
            predicted_laue_spots, distances, orientation_matrix, point1_idx, point2_idx, hkl1, hkl2, coord1, coord2, grain1, grain2 = get_laue_pattern(
                material, point_heuristic_dict, confidences, all_hkl, cluster_points, cluster_points_idx,
                cid, laue_spots, s_grain_id, max_distance_threshold, always_include_center, data_config
            )
        except:
            all_number_explained_points.append(0)
            continue

        number_explained_points = len(list(np.where(distances.min(axis=1) < max_distance_threshold)[0]))
        all_number_explained_points.append(number_explained_points)

    #best_cluster_idx = CLUSTER_HEURISTICS[heuristic](all_confidences)
    best_cluster_idx = np.argmax(all_number_explained_points)

    best_cluster = clusters[best_cluster_idx]
    best_all_confidences = all_confidences[best_cluster_idx]
    best_correct_idx = correct_idxs[best_cluster_idx]
    best_hkl = all_hkls[best_cluster_idx]

    best_cluster_points_idx = np.array(list(best_cluster))
    best_cluster_points = laue_spots[best_cluster_points_idx]

    return best_cluster, best_cluster_points_idx, best_cluster_points, best_all_confidences, best_correct_idx, best_hkl, all_y_true, all_y_pred, all_number_explained_points[best_cluster_idx]


def compute_laue_pattern(hkl1, coord1, hkl2, coord2, B, material, data_config):
    # TODO: Compute the complete pattern corresponding to the best cluster
    orientation_matrix = OrientMatrix_from_2hkl(hkl1, coord1, hkl2, coord2, B)

    # Generate grain with computed orientation
    grain = CP.Prepare_Grain(material, orientation_matrix)

    # Compute coordinates and energies
    # TODO: change hardcoded parameters to material parameters?
    s_tth_predicted, s_chi_predicted, _, _, _, _ = LT.SimulateLaue_full_np(
        grain, data_config["emin"], data_config["emax"], data_config["detectorparameters"],
        pixelsize=data_config["pixelsize"], detectordiameter=300, dim=(data_config["dim1"], data_config["dim2"])
        #5, 22, [79.553, 979.32, 932.31, 0.37, 0.447], pixelsize=0.0734, detectordiameter=300, dim=(2018, 2016)
    )

    # Compute distance between observed and predicted points
    predicted_laue_spots = np.vstack((s_tth_predicted, s_chi_predicted)).T
    predicted_laue_spots = np.unique(predicted_laue_spots, axis=0)

    return predicted_laue_spots, orientation_matrix


def get_laue_pattern(material, point_heuristic_dict, confidences, hkl, cluster_points, best_cluster_points_idx, cid, laue_spots,
                     s_grain_id, max_distance, always_include_center=False, data_config=None):
    lattice_params = dictLT.dict_Materials[material][1]
    B = CP.calc_B_RR(lattice_params)

    candidate_idx = point_heuristic(confidences, hkl, **point_heuristic_dict)
    hkl_changes = np.array(list(itertools.product([1, -1], repeat=3)))
    n_explained_points = np.zeros((len(candidate_idx), len(candidate_idx), len(hkl_changes), len(hkl_changes)))

    for i in range(0, len(candidate_idx)):
        for j in range(i + 1, len(candidate_idx)):
            idx1 = candidate_idx[i]
            idx2 = candidate_idx[j]
            hkl1, hkl2 = hkl[idx1], hkl[idx2]
            if np.array_equal(hkl1, hkl2):
                continue
            coord1, coord2 = cluster_points[idx1], cluster_points[idx2]

            for k1, hkl_change1 in enumerate(hkl_changes):
                for k2, hkl_change2 in enumerate(hkl_changes):
                    hkl1_changed = hkl1 * hkl_change1
                    hkl2_changed = hkl2 * hkl_change2
                    if np.array_equal(hkl1_changed, hkl2_changed) or np.array_equal(hkl1_changed, -hkl2_changed):
                        continue

                    orientation_matrix = OrientMatrix_from_2hkl(hkl1_changed, coord1, hkl2_changed, coord2, B, frame='lauetools')

                    # Check if the orientation matrix is valid
                    if not np.allclose(orientation_matrix @ orientation_matrix.T, np.eye(3), atol=0.01):
                        continue
                    if not np.isclose(np.linalg.det(orientation_matrix), 1, atol=0.01):
                        continue

                    predicted_laue_spots, orientation_matrix = compute_laue_pattern(hkl1_changed, coord1, hkl2_changed, coord2, B, material, data_config)
                    distances = scipy.spatial.distance.cdist(laue_spots, predicted_laue_spots)

                    # Select all points that are very close to a predicted Laue spot to be removed
                    explained_idx = np.where(distances.min(axis=1) < max_distance)[0]
                    n_explained_points[i][j][k1][k2] = len(explained_idx)

    assert n_explained_points.sum() > 0, "No points are explained."
    i, j, k1, k2 = np.unravel_index(np.argmax(n_explained_points, axis=None), n_explained_points.shape)
    if always_include_center and i != 0:
        i = 0
    point1_idx = candidate_idx[i]
    point2_idx = candidate_idx[j]
    hkl1, hkl2 = hkl[point1_idx], hkl[point2_idx]
    coord1, coord2 = cluster_points[point1_idx], cluster_points[point2_idx]
    grain1, grain2 = s_grain_id[best_cluster_points_idx[point1_idx]], s_grain_id[best_cluster_points_idx[point2_idx]]

    hkl1 = hkl1 * hkl_changes[k1]
    hkl2 = hkl2 * hkl_changes[k2]

    predicted_laue_spots, orientation_matrix = compute_laue_pattern(hkl1, coord1, hkl2, coord2, B, material, data_config)
    distances = scipy.spatial.distance.cdist(laue_spots, predicted_laue_spots)

    return predicted_laue_spots, distances, orientation_matrix, point1_idx, point2_idx, hkl1, hkl2, coord1, coord2, grain1, grain2


def confusing_y_axis_check(hkl1, coord1, hkl2, coord2, true_orientation, material):
    lattice_params = dictLT.dict_Materials[material][1]
    B = CP.calc_B_RR(lattice_params)
    understand_matorient_convention(hkl1, coord1, hkl2, coord2, B, true_orientation)


def get_true_laue_pattern(s_grain_id, best_cluster_points_idx, data, material, data_config):
    c_list = [s_grain_id[i] for i in best_cluster_points_idx]
    main_grain_id = int(max(set(c_list), key=c_list.count))
    true_orientation = data['ori_mat'].squeeze(0)[main_grain_id]
    true_grain = CP.Prepare_Grain(material, true_orientation)
    true_tth, true_chi, _, _, _, _ = LT.SimulateLaue_full_np(
        true_grain, data_config["emin"], data_config["emax"], data_config["detectorparameters"],
        pixelsize=data_config["pixelsize"], detectordiameter=300, dim=(data_config["dim1"], data_config["dim2"])
    )
    true_laue_spots = np.vstack((true_tth, true_chi)).T
    true_laue_spots = np.unique(true_laue_spots, axis=0)

    return true_laue_spots, true_tth, true_chi, true_orientation, main_grain_id


def compute_all_true_laue_patterns(data, material, data_config):
    all_true_laue_patterns = []

    for true_orientation in data['ori_mat'].squeeze(0):
        true_grain = CP.Prepare_Grain(material, true_orientation)
        true_tth, true_chi, _, _, _, _ = LT.SimulateLaue_full_np(
            true_grain, data_config["emin"], data_config["emax"], data_config["detectorparameters"],
            pixelsize=data_config["pixelsize"], detectordiameter=300, dim=(data_config["dim1"], data_config["dim2"])
        )
        true_laue_spots = np.vstack((true_tth, true_chi)).T
        true_laue_spots = np.unique(true_laue_spots, axis=0)
        all_true_laue_patterns.append(true_laue_spots)

    return all_true_laue_patterns


def update_data(distances, max_distance, s_tth, s_chi, location, s_grain_id, laue_spots):
    idx_to_remove = np.where(distances.min(axis=1) < max_distance)[0]

    idx_to_keep = list(range(len(s_tth)))
    idx_to_keep = [x for x in idx_to_keep if x not in idx_to_remove]

    explained_points = {}
    explained_points["s_tth"] = s_tth[idx_to_remove]
    explained_points["s_chi"] = s_chi[idx_to_remove]
    explained_points["location"] = location[idx_to_remove]
    explained_points["s_grain_id"] = s_grain_id[idx_to_remove]
    explained_points["laue_spots"] = laue_spots[idx_to_remove]

    s_tth = s_tth[idx_to_keep]
    s_chi = s_chi[idx_to_keep]
    location = location[idx_to_keep]
    s_grain_id = s_grain_id[idx_to_keep]
    laue_spots = laue_spots[idx_to_keep]

    return s_tth, s_chi, location, s_grain_id, laue_spots, idx_to_remove, idx_to_keep, explained_points


def chamfer_distance(predicted_laue_spots, true_laue_spots):
    tree1 = cKDTree(predicted_laue_spots)
    tree2 = cKDTree(true_laue_spots)

    # nearest neighbor distances
    dist1, _ = tree1.query(true_laue_spots)
    dist2, _ = tree2.query(predicted_laue_spots)

    # symmetric chamfer distance
    return np.mean(dist1**2) + np.mean(dist2**2)


def compute_cd_matrix(predicted_laue_spots, all_true_laue_patterns, k=10):
    chamfer_distances = []
    d = len(all_true_laue_patterns)
    for j in range(k):
        chamfer_dists = []
        if j >= len(predicted_laue_spots):
            chamfer_distances.append(np.full((d), 99.99))
            continue
        for true_laue_pattern in all_true_laue_patterns:
            chamfer_dist = chamfer_distance(predicted_laue_spots[j], true_laue_pattern)
            chamfer_dists.append(chamfer_dist.item())
        chamfer_distances.append(chamfer_dists)

    return np.array(chamfer_distances)


def compute_relevance_scores(predicted_laue_spots, all_true_laue_patterns, k=10, eps=1e-5):
    chamfer_distances = compute_cd_matrix(predicted_laue_spots, all_true_laue_patterns, k)
    relevance_scores = (chamfer_distances < eps).any(axis=1).astype(int)
    return relevance_scores, chamfer_distances


def average_precision_at_k(predicted_laue_spots, all_true_laue_patterns, k=10, eps=1e-5):
    relevance_scores,_ = compute_relevance_scores(predicted_laue_spots, all_true_laue_patterns, k, eps)
    relevance_idx = np.where(relevance_scores == 1)[0]

    if len(relevance_idx) == 0:
        return 0.

    precisions = [np.sum(relevance_scores[:i+1]) / (i+1) for i in relevance_idx]
    average_precision = np.sum(precisions) / np.sum(relevance_scores)

    return average_precision


def ndcg_at_k(predicted_laue_spots, all_true_laue_patterns, k=10, eps=1e-5):
    relevance_scores,_ = compute_relevance_scores(predicted_laue_spots, all_true_laue_patterns, k, eps)

    # DCG
    ranks = np.arange(1, len(relevance_scores) + 1)
    discounts = np.log2(ranks + 1)
    dcg = np.sum(relevance_scores / discounts)

    # IDCG
    ideal_relevance_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(ideal_relevance_scores / discounts)

    if idcg == 0:
        return 0.

    ndcg = dcg / idcg

    return ndcg


def precision_at_k(predicted_laue_spots, all_true_laue_patterns, k=10, eps=1e-5):
    relevance_scores,_ = compute_relevance_scores(predicted_laue_spots, all_true_laue_patterns, k, eps)
    precision = np.sum(relevance_scores) / len(relevance_scores)
    return precision


def recall_at_k(predicted_laue_spots, all_true_laue_patterns, k=10, eps=1e-5):
    relevance_scores,_ = compute_relevance_scores(predicted_laue_spots, all_true_laue_patterns, k, eps)
    recall = np.sum(relevance_scores) / len(all_true_laue_patterns)
    return recall
