import warnings

import numpy as np


def cluster_heuristic_mean_confidence(all_confidences, *args):
    mean_confidences = [np.mean(conf) for conf in all_confidences]
    best_cluster_idx = np.argmax(mean_confidences)
    return best_cluster_idx


def cluster_heuristic_top_k_confidence(all_confidences, k=5):
    top_confidences = np.array([sorted(conf, reverse=True)[:k] if len(conf) >= k else np.zeros(k) for conf in all_confidences])
    # try:
    best_cluster_idx = np.argmax(np.mean(top_confidences, axis=1))
    # except:
    #     raise ValueError("Something wrong wrt best_cluster_idx, returning zero not to kill things?")
    return best_cluster_idx


def point_heuristic(confidences, miller_idx, k=None, confidence_threshold=0, sort_by='miller'):
    if k is None:
        k = len(confidences)
    elif k > len(confidences):
        warnings.warn(f"Can only select {len(confidences)} instead of k={k} points.")

    # Only consider sufficiently confident points
    confident_indices = np.where(confidences > confidence_threshold)[0]
    confident_miller_idx = miller_idx[confident_indices]
    confident_confidences = confidences[confident_indices]

    # Sort confident points according to the selected method
    if sort_by is None:
        sorted_idx = np.arange(len(confident_confidences))
    elif sort_by in ['miller', 'hkl']:
        sorted_idx = np.argsort(np.linalg.norm(confident_miller_idx, axis=1))
    elif sort_by == 'confidence':
        sorted_idx = np.argsort(confident_confidences)[::-1]
    else:
        raise ValueError("Unknown sort method: " + sort_by)

    # Select and return the top-k points
    top_idx = confident_indices[sorted_idx][:k]
    return top_idx
