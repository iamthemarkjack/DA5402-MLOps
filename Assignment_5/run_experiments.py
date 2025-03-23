import os
import subprocess
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/run_experiments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_experiments")

def update_params(dataset_version, seed):
    """
    Update params.yaml with the given dataset version and seed
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    params['dataset_version'] = dataset_version
    params['random_seed'] = seed
    
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)
    
    logger.info(f"Updated params.yaml with dataset_version={dataset_version}, random_seed={seed}")

def run_experiment(dataset_version, seed):
    """
    Run a single experiment with the given dataset version and seed
    """
    update_params(dataset_version, seed)
    
    exp_name = f"{dataset_version.replace('+', '_plus_')}_seed_{seed}"
    
    # run DVC experiment
    logger.info(f"Running experiment: {exp_name}")
    subprocess.run([
        "dvc", "exp", "run", "--name", exp_name
    ], check=True)
    
    logger.info(f"Experiment {exp_name} completed")

def main():
    # dataset versions to run
    dataset_versions = ["v1", "v2", "v3", "v1+v2", "v1+v2+v3"]
    
    # random seeds
    seeds = [26, 45, 69]
    
    # run all combinations
    for dataset_version in dataset_versions:
        for seed in seeds:
            run_experiment(dataset_version, seed)
    
    # show experiments
    logger.info("All experiments completed. Showing experiment results:")
    subprocess.run(["dvc", "exp", "show"], check=True)

if __name__ == "__main__":
    main()