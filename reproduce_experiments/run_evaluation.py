import argparse
import torch
from ruamel.yaml import YAML
from evaluate_clustering import evaluate_model
from reproduce_experiments.utils.general_utils import make_reproducible


def main(
    data_config, model_config, clustering_config, data_directory=None, verbose_eval=False, shuffle=False,
    give_us_all=False, give_us_some=True, data_interval=None, slow_mode=False
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Configurations:")
    print(f"Data config: {data_config}")
    print(f"Model config: {model_config}")
    print(f"Clustering config: {clustering_config}")

    yaml = YAML()
    with open(data_config) as f:
        data_config = yaml.load(f)
    with open(model_config) as f:
        model_config = yaml.load(f)
    with open(clustering_config) as f:
        clustering_config = yaml.load(f)

    make_reproducible(42)

    evaluate_model(
        data_directory, data_config, model_config, clustering_config, device, verbose_eval=verbose_eval,
        shuffle=shuffle, give_us_all=give_us_all, give_us_some=give_us_some, slow_mode=slow_mode,
        data_interval=data_interval
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=False, default='../configurations/materials/Cu-10.yaml')
    parser.add_argument("--model_config", required=False, default='../configurations/models/mlp_10grain.yaml')
    parser.add_argument("--clustering_config", required=False, default='../configurations/clustering/config.yaml')
    parser.add_argument("--data_directory", required=False, default=None)
    parser.add_argument('--verbose_eval', action='store_true')
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--give_us_all", action='store_true')
    parser.add_argument("--give_us_some", action='store_true')
    parser.add_argument("--data_interval", required=False, default=None, nargs=2, type=int)
    parser.add_argument("--slow_mode", action='store_true')
    args = parser.parse_args()

    # Check arguments
    if args.give_us_all and args.give_us_some:
        raise ValueError("give_us_all and give_us_some cannot be both True.")

    main(
        args.data_config, args.model_config, args.clustering_config, args.data_directory, args.verbose_eval,
        args.shuffle, args.give_us_all, args.give_us_some, args.data_interval, args.slow_mode
    )
