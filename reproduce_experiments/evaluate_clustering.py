import csv
from datetime import datetime

import matplotlib.pyplot as plt
from utils.clustering_utils import *
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from reproduce_experiments.utils.clustering_utils import compute_gnomonic_projection
from reproduce_experiments.utils.general_utils import CodeTimer


def evaluate_clusters(
        n_clusters, n_bins, star_center_count, n_lines, material, ang_bins, classhkl, cluster_heuristic,
        point_heuristic_dict, confidence_threshold, max_distance_threshold, data, s_tth, s_chi, location, s_grain_id,
        laue_spots, model, verbose=False, device=torch.device('cuda:0'), always_include_center=False, data_config=None,
        slow_mode=False
):
    # Collect data for evaluation metrics
    explained_points_total = 0
    explained_points_per_grain = {}
    y_preds, y_trues = None, None
    predicted_laue_spots_all, true_laue_spots_all = [], []
    true_grain_ids = []

    results_dict = {}

    bad_cluster_idxs = []

    for cid in range(n_clusters):
        points = compute_gnomonic_projection(s_tth, s_chi)
        if verbose:
            print(f"Cluster id: {cid}")
            print(f"Number of points left: {len(points)}")
        if len(points) < 4:  # TODO: somehow 2 points break stuff sometime.
            if verbose:
                print("Too few points to cluster.")
            break

        logger = None
        if verbose:
            logger = print

        with CodeTimer(f"Hough embedding, fast? {cid}", logger=logger):
            edge_embeddings, colors, edge_to_points, h, angle_values, distance_values = compute_hough_embeddings(
                points, s_grain_id, n_bins
            )
        with CodeTimer(f"Star centers, fast? {cid}", logger=logger):
            star_center_idx, point_to_lines_idx = compute_star_centers(
                star_center_count, n_lines, h, angle_values, distance_values, edge_to_points, edge_embeddings
            )
        # print(f"Number of star centers: {len(star_center_idx)}")
        if len(star_center_idx) < 1:
            if verbose:
                print("No star centers found.")
            break
        with CodeTimer(f"Cluster candidates, {cid}", logger=logger):
            cluster_candidates = compute_cluster_candidates(star_center_idx, point_to_lines_idx, cid)

        if bad_cluster_idxs is not None:
            if len(bad_cluster_idxs) > 0:
                for bid in bad_cluster_idxs:
                    cluster_candidates = np.delete(cluster_candidates, bid)

        if len(cluster_candidates) < 1:
            if verbose:
                print("No clusters found.")
            break

        with CodeTimer(f"Get candidates, {cid}", logger=logger):
            best_cluster, best_cluster_points_idx, best_cluster_points, best_confidences, best_correct_idx, best_hkl, best_y_trues, best_y_preds, best_cluster_idx = get_cluster(
                cluster_candidates, laue_spots, ang_bins, classhkl, location, cluster_heuristic, model, device, bad_cluster_idxs=bad_cluster_idxs
            )

        # Stopping criteria
        if np.mean(best_confidences) < confidence_threshold:
            if verbose:
                print("Best mean confidence at iteration " + str(cid + 1) + " too low.")
            break

        # Get predicted Laue spots. If no points are explained, continue.
        try:
            with CodeTimer(f"Get Laue Patterns, {cid}", logger=logger):
                predicted_laue_spots, distances, orientation_matrix, point1_idx, point2_idx, hkl1, hkl2, coord1, coord2, grain1, grain2 = get_laue_pattern(
                    material, point_heuristic_dict, best_confidences, best_hkl, best_cluster_points, best_cluster_points_idx, cid,
                    laue_spots, s_grain_id, max_distance_threshold, always_include_center, data_config)
            bad_cluster = False
        except AssertionError:
            if verbose:
                print("No points are explained, breaking.")
            if slow_mode:
                bad_cluster = True
            else:
                break

        if bad_cluster:
            bad_cluster_idxs.append(best_cluster_idx)
            continue
        else:
            bad_cluster_idxs = []

        # Get true Laue spots
        with CodeTimer(f"Get true Laue Patterns, {cid}", logger=logger):
            true_laue_spots, true_tth, true_chi, true_orientation, true_grain = get_true_laue_pattern(
                s_grain_id, best_cluster_points_idx, data, material, data_config
            )

        predicted_laue_spots_all.append(predicted_laue_spots)
        true_laue_spots_all.append(true_laue_spots)
        true_grain_ids.append(true_grain)

        # Update data
        s_tth, s_chi, location, s_grain_id, laue_spots, idx_to_remove, idx_to_keep, explained_points_dict = update_data(
            distances, max_distance_threshold, s_tth, s_chi, location, s_grain_id, laue_spots
        )

        # Storing data for accuracy metrics
        if y_preds is None:
            y_preds = best_y_preds
        else:
            y_preds = np.concatenate((y_preds, best_y_preds), axis=0)

        if y_trues is None:
            y_trues = best_y_trues
        else:
            y_trues = np.concatenate((y_trues, best_y_trues), axis=0)

        # Storing data for other metrics
        explained_points_total += len(idx_to_remove)

        for idx in explained_points_dict["s_grain_id"]:
            explained_points_per_grain[int(idx)] = explained_points_per_grain.get(int(idx), 0) + 1

    all_true_laue_patterns = compute_all_true_laue_patterns(data, material, data_config)

    results_dict["predicted_laue_spots"] = predicted_laue_spots_all
    results_dict["true_laue_spots"] = true_laue_spots_all
    results_dict["all_true_laue_patterns"] = all_true_laue_patterns
    results_dict["true_grain_ids"] = true_grain_ids
    results_dict["y_preds"] = y_preds
    results_dict["y_trues"] = y_trues
    results_dict["explained_points_total"] = explained_points_total
    results_dict["explained_poinst_per_grain"] = explained_points_per_grain
    results_dict["remaining_grains"] = len(np.unique(s_grain_id))

    return results_dict


def evaluate_model(
        data_directory, data_config: dict, model_config: dict, clustering_config: dict, device: torch.device=None,
        verbose=False, verbose_eval=False, give_us_all=True, shuffle=False, give_us_some=True, slow_mode=False,
        data_interval=None
):
    if device is None:
        device = torch.device("cuda:0")

    dataloader, classhkl, ang_bins, class_filter, material, n_rotations = load_test_data(
        data_config, data_directory, give_us_all=give_us_all, shuffle=shuffle, give_us_some=give_us_some
    )
    model = load_model(data_config, model_config, len(ang_bins), len(classhkl), device)
    model.eval()

    # setup clustering parameters
    n_lines = clustering_config.get("n_lines", None)  # number of lines to be detected for clustering
    n_clusters = clustering_config.get("n_clusters", 100)  # maximum number of clusters
    n_bins = clustering_config.get("n_bins", 500)  # number of bins for the histogram
    star_center_count = clustering_config.get("star_center_count", 3)  # minimum number of intersecting lines to define a star center
    confidence_threshold = clustering_config.get("confidence_threshold", 0.8)  # early stopping criteria based on best_mean_confidence
    max_distance_threshold = clustering_config.get("max_distance", 0.2)  # tolerance for explaining points based on computed points
    cluster_heuristic = clustering_config.get("cluster_heuristic", "mean_confidence") #TODO
    point_heuristic_dict = clustering_config.get("point_heuristic", {}) #TODO

    # Cheat codes
    remove_only_true_predictions = False
    perfect_cluster = False  # sets the computed clusters to perfectly match grains
    always_include_center = False  # sets first/second point of selected pair to be center point

    all_accuracies = []
    all_explained_points_ratios = []
    all_explained_grains_ratios = []
    all_precisions = {}
    all_recalls = {}
    all_aps = {}
    all_ndcgs = {}
    all_relevance_scores = []
    runtimes = []
    n_spots = []

    ks = [1,3,5,10,20]

    for k in ks:
        all_precisions[k] = []
        all_recalls[k] = []
        all_aps[k] = []
        all_ndcgs[k] = []

    if data_interval is None:
        data_interval = (0, len(dataloader))
    n_iterations = data_interval[1] - data_interval[0]

    # Evaluation.
    pbar = tqdm(desc='Run evaluation', total=n_iterations)
    for i, (data, s_tth, s_chi, location, s_grain_id, s_miller_ind, laue_spots, file_idx) in enumerate(dataloader):
        s_tth, s_chi, location, s_grain_id, laue_spots = s_tth[0].numpy(), s_chi[0].numpy(), location[0].numpy(), s_grain_id[0].numpy(), laue_spots[0].numpy()

        if i < data_interval[0] or i >= data_interval[1]:
            continue

        if n_lines is None:
            n_lines = 10 * len(np.unique(s_grain_id))

        n_spots.append(len(s_tth))

        logger = None
        if verbose:
            logger = print

        with CodeTimer(f"Cluster evaluation, datapoint {i}", store=runtimes, logger=logger):
            results = evaluate_clusters(
                n_clusters, n_bins, star_center_count, n_lines, material, ang_bins, classhkl, cluster_heuristic,
                point_heuristic_dict, confidence_threshold, max_distance_threshold, data, s_tth, s_chi, location,
                s_grain_id, laue_spots, model, verbose, device, always_include_center, data_config,
                slow_mode
            )

        predicted_laue_spots = results["predicted_laue_spots"]
        true_laue_spots = results["true_laue_spots"]
        all_true_laue_patterns = results["all_true_laue_patterns"]
        y_preds = results["y_preds"]
        y_trues = results["y_trues"]
        true_grain_ids = results["true_grain_ids"]
        explained_points_total = results["explained_points_total"]
        explained_points_per_grain = results["explained_poinst_per_grain"]
        remaining_grains = results["remaining_grains"]

        # Evaluation metrics
        # Accuracy
        if y_preds is None or y_trues is None:
            accuracy = 0.
        else:
            accuracy = accuracy_score(y_trues, y_preds)
        all_accuracies.append(accuracy)

        # Explained points ratio
        explained_points_ratio = explained_points_total / len(s_tth)
        all_explained_points_ratios.append(explained_points_ratio)

        # Explained grains ratio
        total_points_per_grain = {}
        for idx in s_grain_id:
            total_points_per_grain[int(idx)] = total_points_per_grain.get(int(idx), 0) + 1

        explained_grain_ratios = []
        for key, value in total_points_per_grain.items():
            explained_grain_ratio = explained_points_per_grain.get(key, 0) / value
            explained_grain_ratios.append(explained_grain_ratio)

        all_explained_grains_ratios.append(sum(g > 0.5 for g in explained_grain_ratios) / len(np.unique(s_grain_id)))

        relevance_scores, chamfer_distances = compute_relevance_scores(
            predicted_laue_spots, all_true_laue_patterns, k=len(all_true_laue_patterns)
        )
        all_relevance_scores.append(relevance_scores)

        for k in ks:
            # Precision@k and Recall@k
            precision = precision_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            recall = recall_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)

            # Average Precision@k
            ap = average_precision_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            all_aps[k].append(ap)

            # NDCG@k
            ndcg = ndcg_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            all_ndcgs[k].append(ndcg)

        if verbose_eval:
            print(f"\nGrains remaining: {remaining_grains}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Explained Points: {explained_points_ratio:.2%}")
            print(f"Explained Grains: {sum(g > 0.5 for g in explained_grain_ratios) / len(np.unique(s_grain_id)):.2%}")
            for key, value in total_points_per_grain.items():
                explained_grain_ratio = explained_points_per_grain.get(key, 0) / value
                print(f"Grain {key:02d}: {explained_grain_ratio:.2%}")

        pbar.update(1)
    pbar.close()

    k = 10

    print(f"Average number of spots per image: {int(np.mean(n_spots))}")
    print(f"Average accuracies: {np.mean(all_accuracies):.3f}")
    print(f"Average explained points ratios: {np.mean(all_explained_points_ratios):.3f}")
    print(f"Average explained grains ratios: {np.mean(all_explained_grains_ratios):.3f}")
    # Retrieval Metrics
    print(f"Precision@{k}: {np.mean(all_precisions[k]):.3f}")
    print(f"Recall@{k}: {np.mean(all_recalls[k]):.3f}")
    print(f"MAP@{k}: {np.mean(all_aps[k]):.3f}")
    print(f"NDCG@{k}: {np.mean(all_ndcgs[k]):.3f}")
    print(f"Average per-pattern ({n_rotations}-grain) runtime: {np.mean(runtimes):.2f}s")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Plot mean relevance scores across rank
    mean_relevance_scores = np.mean(np.array(all_relevance_scores), axis=0)
    x = np.arange(1,len(all_true_laue_patterns)+1)
    plt.plot(x, mean_relevance_scores, marker='o', linestyle='-')
    plt.xticks(x)
    plt.ylim(0,1)
    plt.xlabel('Rank')
    plt.ylabel('Relevance')
    plt.title('Mean Relevance per Rank')
    plt.show()

    model_n_rotations = model_config.get("n_rotations", n_rotations)

    os.makedirs('results', exist_ok=True)
    with open(f"results/{material}_{n_rotations}_m{model_n_rotations}_{str(data_interval)}_slowMode{str(slow_mode)}_{timestamp}.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        cols = [
            "avg_n_spots", "avg_acc", "avg_explained_points_ratio", "avg_explained_grains_ratio","avg_image_runtime",
            "mean_relevance_scores"
        ]
        for k in ks:
            cols.extend(["precision@" + str(k), "recall@" + str(k), "MAP@" + str(k), "NDCG@" + str(k)])
        writer.writerow(cols)
        values = [
            int(np.mean(n_spots)), np.mean(all_accuracies), np.mean(all_explained_points_ratios),
            np.mean(all_explained_grains_ratios), np.mean(runtimes), mean_relevance_scores
        ]
        for k in ks:
            values.extend([np.mean(all_precisions[k]), np.mean(all_recalls[k]), np.mean(all_aps[k]), np.mean(all_ndcgs[k])])
        writer.writerow(values)
        #writer.writerow([
        #    int(np.mean(n_spots)), np.mean(all_accuracies), np.mean(all_explained_points_ratios),
        #    np.mean(all_explained_grains_ratios), np.mean(all_precisions), np.mean(all_recalls), np.mean(all_aps),
        #    np.mean(all_ndcgs), np.mean(runtimes)
        #])

    return True


def evaluate_baseline(data_directory, data_config: dict, model_config: dict, clustering_config: dict,
                      device: torch.device=torch.device("cuda:0"), verbose=False, verbose_summary=False,
                      verbose_eval=False, give_us_all=True, shuffle=False, give_us_some=True, noisy=False):
    dataloader, classhkl, ang_bins, class_filter, material, n_rotations = load_test_data(data_config, data_directory, give_us_all=give_us_all, shuffle=shuffle, give_us_some=give_us_some)
    model = load_model(data_config, model_config, len(ang_bins), len(classhkl), device)
    model.eval()

    # Setting up stuff.
    all_accuracies = []
    all_explained_points_ratios = []
    all_explained_grains_ratios = []
    all_precisions = {}
    all_recalls = {}
    all_aps = {}
    all_ndcgs = {}
    all_relevance_scores = []
    runtimes = []
    n_spots = []

    ks = [1,3,5,10,20]

    for k in ks:
        all_precisions[k] = []
        all_recalls[k] = []
        all_aps[k] = []
        all_ndcgs[k] = []

    # Parameter setup.
    n_clusters = clustering_config.get("n_clusters", 100)  # maximum number of clusters
    confidence_threshold = clustering_config.get("confidence_threshold", 0.8)  # early stopping criteria based on best_mean_confidence
    max_distance_threshold = clustering_config.get("max_distance", 0.2)  # tolerance for explaining points based on computed points
    max_number_candidates = clustering_config.get("max_number_candidates", 100)
    cluster_heuristic = clustering_config.get("cluster_heuristic", "mean_confidence") #TODO

    # Evaluation.
    # data_interval = None
    data_interval = [0, 10]
    if noisy:
        data_interval = [150, 160]
    timings = []

    for i, (data, s_tth, s_chi, location, s_grain_id, s_miller_ind, laue_spots, file_idx) in enumerate(tqdm(dataloader)):
        s_tth, s_chi, location, s_grain_id, laue_spots = s_tth[0].numpy(), s_chi[0].numpy(), location[0].numpy(), s_grain_id[0].numpy(), laue_spots[0].numpy()

        if data_interval is not None:
            if i < data_interval[0] or i >= data_interval[1]:
                continue

        # print(f"\nTotal number of spots: {len(s_tth)}.")
        n_spots.append(len(s_tth))

        logger = None
        if verbose:
            logger = print

        with CodeTimer(f"Baseline, image {i}", logger=logger, store=timings):
            results = evaluate_full_pattern(
                n_clusters, max_number_candidates, material, ang_bins, classhkl, cluster_heuristic,
                confidence_threshold, max_distance_threshold, data, s_tth, s_chi, location,
                s_grain_id, laue_spots, model, verbose, verbose_summary, device, False, data_config
            )

        predicted_laue_spots = results["predicted_laue_spots"]
        true_laue_spots = results["true_laue_spots"]
        all_true_laue_patterns = results["all_true_laue_patterns"]
        y_preds = results["y_preds"]
        y_trues = results["y_trues"]
        true_grain_ids = results["true_grain_ids"]
        explained_points_total = results["explained_points_total"]
        explained_points_per_grain = results["explained_points_per_grain"]
        remaining_grains = results["remaining_grains"]

        # Evaluation metrics
        # Accuracy
        if y_preds is None or y_trues is None:
            accuracy = 0.
        else:
            accuracy = accuracy_score(y_trues, y_preds)
        all_accuracies.append(accuracy)

        # Explained points ratio
        explained_points_ratio = explained_points_total / len(s_tth)
        all_explained_points_ratios.append(explained_points_ratio)

        # Explained grains ratio
        total_points_per_grain = {}
        for idx in s_grain_id:
            total_points_per_grain[int(idx)] = total_points_per_grain.get(int(idx), 0) + 1

        explained_grain_ratios = []
        for key, value in total_points_per_grain.items():
            explained_grain_ratio = explained_points_per_grain.get(key, 0) / value
            explained_grain_ratios.append(explained_grain_ratio)

        explained_grain_ratio = sum(g > 0.5 for g in explained_grain_ratios) / len(np.unique(s_grain_id))

        all_explained_grains_ratios.append(explained_grain_ratio)

        relevance_scores, _ = compute_relevance_scores(
            predicted_laue_spots, all_true_laue_patterns, k=len(all_true_laue_patterns)
        )
        all_relevance_scores.append(relevance_scores)

        for k in ks:
            # Precision@k and Recall@k
            precision = precision_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            recall = recall_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)

            # Average Precision@k
            ap = average_precision_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            all_aps[k].append(ap)

            # NDCG@k
            ndcg = ndcg_at_k(predicted_laue_spots, all_true_laue_patterns, k=k)
            all_ndcgs[k].append(ndcg)

        if verbose_eval:
            print(f"\nGrains remaining: {remaining_grains}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Explained Points: {explained_points_ratio:.2%}")
            print(f"Explained Grains: {explained_grain_ratio:.2%}")
            for key, value in total_points_per_grain.items():
                explained_grain_ratio = explained_points_per_grain.get(key, 0) / value
                print(f"Grain {key:02d}: {explained_grain_ratio:.2%}")

        #if i > 5:
        #    break

    print(f"Average runtime per-image: {np.mean(timings)}s")

    k = 10

    print(f"Average number of spots per image: {int(np.mean(n_spots))}")
    print(f"Average accuracies: {np.mean(all_accuracies):.3f}")
    print(f"Average explained points ratios: {np.mean(all_explained_points_ratios):.3f}")
    print(f"Average explained grains ratios: {np.mean(all_explained_grains_ratios):.3f}")
    # Retrieval Metrics
    print(f"Precision@{k}: {np.mean(all_precisions[k]):.3f}")
    print(f"Recall@{k}: {np.mean(all_recalls[k]):.3f}")
    print(f"MAP@{k}: {np.mean(all_aps[k]):.3f}")
    print(f"NDCG@{k}: {np.mean(all_ndcgs[k]):.3f}")
    print(f"Average per-pattern ({n_rotations}-grain) runtime: {np.mean(timings):.2f}s")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Plot mean relevance scores across rank
    mean_relevance_scores = np.mean(np.array(all_relevance_scores), axis=0)
    x = np.arange(1,len(all_true_laue_patterns)+1)
    plt.plot(x, mean_relevance_scores, marker='o', linestyle='-')
    plt.xticks(x)
    plt.ylim(0,1)
    plt.xlabel('Rank')
    plt.ylabel('Relevance')
    plt.title('Mean Relevance per Rank')
    plt.show()

    with open(f"results/{material}_unfiltered_{give_us_some}_noisy_{noisy}_{timestamp}.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        cols = [
            "avg_n_spots", "std_n_spots",
            "avg_acc", "std_acc",
            "avg_explained_points_ratio", "std_explained_points_ratio",
            "avg_explained_grains_ratio", "std_explained_grains_ratio",
            "avg_image_runtime", "std_image_runtime",
            "mean_relevance_scores", "std_relevance_scores"
        ]
        for k in ks:
            cols.extend(["precision@" + str(k), "std_precision@" + str(k),
                         "recall@" + str(k), "std_recall@" + str(k),
                         "MAP@" + str(k), "std_MAP@" + str(k),
                         "NDCG@" + str(k), "std_NDCG@" + str(k)])
        writer.writerow(cols)
        values = [
            int(np.mean(n_spots)),
            int(np.std(n_spots)),
            np.round(np.mean(all_accuracies), 3),
            np.round(np.std(all_accuracies),3),
            np.round(np.mean(all_explained_points_ratios),3),
            np.round(np.std(all_explained_points_ratios),3),
            np.round(np.mean(all_explained_grains_ratios),3),
            np.round(np.std(all_explained_grains_ratios),3),
            np.round(np.mean(timings),3),
            np.round(np.std(timings),3),
            np.round(mean_relevance_scores,3),
            np.round(np.std(np.array(all_relevance_scores), axis=0), 3),
        ]
        for k in ks:
            values.extend([np.round(np.mean(all_precisions[k]), 3),
                           np.round(np.std(all_precisions[k]), 3),
                           np.round(np.mean(all_recalls[k]), 3),
                           np.round(np.std(all_recalls[k]), 3),
                           np.round(np.mean(all_aps[k]), 3),
                           np.round(np.std(all_aps[k]), 3),
                           np.round(np.mean(all_ndcgs[k]), 3),
                           np.round(np.std(all_ndcgs[k]), 3)])
        writer.writerow(values)
        #writer.writerow([
        #    int(np.mean(n_spots)), np.mean(all_accuracies), np.mean(all_explained_points_ratios),
        #    np.mean(all_explained_grains_ratios), np.mean(all_precisions), np.mean(all_recalls), np.mean(all_aps),
        #    np.mean(all_ndcgs), np.mean(runtimes)
        #])

    return True



def evaluate_full_pattern(n_clusters, max_number_candidates, material, ang_bins, classhkl, cluster_heuristic,
                      confidence_threshold, max_distance_threshold, data, s_tth, s_chi, location, s_grain_id, laue_spots,
                      model, verbose=False, verbose_summary=False, device=torch.device('cuda:0'), always_include_center=False, data_config=None):

    # Collect data for evaluation metrics
    predicted_laue_spots_all, true_laue_spots_all = [], []
    true_grain_ids = []
    results_dict = {}

    total_points_per_grain = {}
    for idx in s_grain_id:
        total_points_per_grain[int(idx)] = total_points_per_grain.get(int(idx), 0) + 1

    logger = None
    if verbose:
        logger = print

    # Getting Miller indices.
    all_confidences = []
    correct_idxs = []
    all_hkls = []
    cluster_points = laue_spots

    # Compute feature vectors only based on one cluster
    X = compute_codebars(cluster_points[:, 0], cluster_points[:, 1], ang_bins, True)

    # Predict labels and compute accuracy
    y_pred = model(torch.tensor(X, dtype=torch.float).to(device)).detach().cpu().numpy()
    y_true = location

    softmax_outputs = scipy.special.softmax(y_pred, axis=1)
    confidences = np.max(softmax_outputs, axis=1)
    correct_idx = y_pred.argmax(axis=1) == y_true
    correct_idxs.append(correct_idx)
    all_confidences.append(confidences)
    all_hkl = classhkl[y_pred.argmax(axis=1)]
    all_hkls.append(all_hkl)
    best_cluster_idx = CLUSTER_HEURISTICS[cluster_heuristic](all_confidences)
    best_all_confidences = all_confidences[best_cluster_idx]
    # best_correct_idx = correct_idxs[best_cluster_idx]
    best_hkl = all_hkls[best_cluster_idx]


    # best_cluster_points = laue_spots
    # best_cluster_points_idx = np.arange(len(laue_spots))
    best_confidences = best_all_confidences

    # For all the points above confidence, compute an orientation matrix
    # and select the one with more explained points

    lattice_params = dictLT.dict_Materials[material][1]
    B = CP.calc_B_RR(lattice_params)

    candidate_idx = np.where(best_confidences > confidence_threshold)[0]
    candidate_idx = candidate_idx[np.argsort(-best_confidences[candidate_idx])]  # Sort by conf. in descending order.
    if max_number_candidates is not None and len(candidate_idx) > max_number_candidates:
        candidate_idx = candidate_idx[:max_number_candidates]

    hkl_changes = np.array(list(itertools.product([1, -1], repeat=3)))

    explained_points_per_grain = {}
    total_explained_idx = []

    for _ in (tqdm(range(n_clusters), desc=f"Processing candidates")):
        n_explained_points = np.zeros((len(candidate_idx), len(candidate_idx), len(hkl_changes), len(hkl_changes)))
        print(f"{len(candidate_idx)} candidates left.")
        if len(candidate_idx) < 2:
            break
        for i in range(0, len(candidate_idx)):
            for j in range(i + 1, len(candidate_idx)):
                idx1 = candidate_idx[i]
                idx2 = candidate_idx[j]
                hkl1, hkl2 = best_hkl[idx1], best_hkl[idx2]
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
                        explained_idx = np.where(distances.min(axis=1) < max_distance_threshold)[0]
                        n_explained_points[i][j][k1][k2] = len(explained_idx)

        # Find the indices of the maximum value to get the max number of explained points.
        max_idx = np.unravel_index(np.argmax(n_explained_points), n_explained_points.shape)
        best_i, best_j, best_k1, best_k2 = max_idx
        max_explained = n_explained_points[best_i, best_j, best_k1, best_k2]
        if max_explained == 0:
            break

        # Get the points in the laue_spot index space.
        sel_id_1 = candidate_idx[best_i]
        sel_id_2 = candidate_idx[best_j]

        # Compute orientation matrix from the best orientation and remove points.
        hkl1_changed, hkl2_changed = best_hkl[sel_id_1] * hkl_changes[best_k1], best_hkl[sel_id_2] * hkl_changes[best_k2]
        coord1, coord2 = cluster_points[sel_id_1], cluster_points[sel_id_2]
        predicted_laue_spots, orientation_matrix = compute_laue_pattern(
            hkl1_changed, coord1, hkl2_changed, coord2, B, material, data_config
        )
        distances = scipy.spatial.distance.cdist(laue_spots, predicted_laue_spots)
        explained_idx = np.where(distances.min(axis=1) < max_distance_threshold)[0]

        total_explained_idx += explained_idx.tolist()

        # Remove explained points from the candidate pool.
        explained_mask = distances.min(axis=1) < max_distance_threshold
        candidate_idx = candidate_idx[~explained_mask[candidate_idx]]

        # Also remove the selected pair themselves.
        candidate_idx = candidate_idx[~np.isin(candidate_idx, np.array([sel_id_1, sel_id_2]))]

        predicted_laue_spots_all.append(predicted_laue_spots)
        true_laue_spots, _, _, _, _ = get_true_laue_pattern(s_grain_id, explained_idx, data, material, data_config)
        true_laue_spots_all.append(true_laue_spots)

        # Can also do the same for laue spots.
        # laue_spots = laue_spots[~explained_mask]

    # Storing data for other metrics
    explained_points_total = len(np.unique(total_explained_idx))

    for idx in s_grain_id[np.unique(total_explained_idx)]:
        explained_points_per_grain[int(idx)] = explained_points_per_grain.get(int(idx), 0) + 1

    all_true_laue_patterns = compute_all_true_laue_patterns(data, material, data_config)
    results_dict["predicted_laue_spots"] = predicted_laue_spots_all
    results_dict["true_laue_spots"] = true_laue_spots_all
    results_dict["all_true_laue_patterns"] = all_true_laue_patterns
    results_dict["true_grain_ids"] = true_grain_ids
    results_dict["y_preds"] = y_pred.argmax(axis=1)
    results_dict["y_trues"] = y_true
    results_dict["explained_points_total"] = explained_points_total
    results_dict["explained_points_per_grain"] = explained_points_per_grain
    results_dict["remaining_grains"] = len(np.unique(s_grain_id))

    return results_dict

