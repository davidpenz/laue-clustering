import argparse

import torch
from ruamel.yaml import YAML
from evaluate_clustering import evaluate_baseline
from reproduce_experiments.utils.general_utils import make_reproducible


def main(data_config, model_config, clustering_config, noisy=False):
    make_reproducible(42)


    yaml = YAML()
    with open(data_config) as f:
        data_config = yaml.load(f)
    with open(model_config) as f:
        model_config = yaml.load(f)
    with open(clustering_config) as f:
        clustering_config = yaml.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    evaluate_baseline(
        None, data_config, model_config, clustering_config, device, verbose_eval=False,
        shuffle=False, give_us_all=False, give_us_some=True, verbose=False, noisy=noisy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=False, default='../configurations/materials/ZrO2-10.yaml')
    parser.add_argument("--model_config", required=False, default='../configurations/models/mlp_basic.yaml')
    parser.add_argument("--clustering_config", required=False, default='../configurations/clustering/baseline.yaml')
    parser.add_argument("--data_directory", required=False, default=None)
    parser.add_argument('--verbose_eval', action='store_true')
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--give_us_all", action='store_true')
    parser.add_argument("--give_us_some", action='store_true')
    parser.add_argument("--noisy", action='store_true')
    args = parser.parse_args()

    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Configurations:")
    print(f"Data config: {args.data_config}")
    print(f"Model config: {args.model_config}")
    print(f"Clustering config: {args.clustering_config}")

    yaml = YAML()
    with open(args.data_config) as f:
        data_config = yaml.load(f)
    with open(args.model_config) as f:
        model_config = yaml.load(f)
    with open(args.clustering_config) as f:
        clustering_config = yaml.load(f)

    data_dir = args.data_directory

    make_reproducible(42)

    evaluate_baseline(
        data_dir, data_config, model_config, clustering_config, device, verbose_eval=args.verbose_eval,
        shuffle=args.shuffle, give_us_all=args.give_us_all, give_us_some=True,
        verbose=False, noisy=args.noisy
        )
