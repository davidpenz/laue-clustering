from run_evaluation import main as run_evaluation
from run_baseline import main as run_baseline

if __name__ == "__main__":
    materials = ["Cu", "Ti", "Sn", "ZrO2"]
    grains = [10]
    data_intervals = [[0,50], [50,100], [100,150], [150,200]]
    slow_modes = [False, True]

    for material in materials:
        for grain in grains:
            for data_interval in data_intervals:
                for slow_mode in slow_modes:
                    data_config = '../configurations/materials/' + material + '-' + str(grain) + '.yaml'
                    model_config = '../configurations/models/mlp_10grain.yaml'
                    clustering_config = '../configurations/clustering/config.yaml'

                    print("Running " + material + str(grain) + " for images " + str(data_interval[0]) + " to " + str(data_interval[1]) + ", slow_mode " + str(slow_mode) + "...")
                    run_evaluation(data_config, model_config, clustering_config, data_interval=data_interval, slow_mode=slow_mode)
                    print("")

    for material in materials:
        for grain in grains:
            for noisy in [True, False]:
                data_config = '../configurations/materials/' + material + '-' + str(grain) + '.yaml'
                model_config = '../configurations/models/mlp_10grain.yaml'
                clustering_config = '../configurations/clustering/baseline.yaml'

                noisy_label = "noisy" if noisy else "not noisy"

                print("Running " + material + str(grain) + " for " + noisy_label +  "  images ")
                run_baseline(data_config, model_config, clustering_config, noisy=noisy)
                print("")

