import os
import glob
import random
import shutil
import argparse
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/create_partitions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("create_partitions")

def create_partition(source_dir, target_dir, sample_size, seed, already_sampled):
    """
    Create a random partition of the dataset by sampling without replacement
    and ensuring no duplicate selections across partitions
    """
    random.seed(seed)

    os.makedirs(target_dir, exist_ok = True)

    # get all images from all classes
    all_images = []
    class_names = next(os.walk(source_dir))[1]
    
    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        images = glob.glob(os.path.join(class_dir, "*.png"))
        all_images.extend(images)

    available_images = list(set(all_images) - already_sampled)

    logger.info(f"Total available images: {len(available_images)}")

    # sample without replacement
    if len(available_images) < sample_size:
        logger.warning(f"Request sample size {sample_size} is larger than available images {len(available_images)}")
        sampled_images = available_images
    else:
        sampled_images = random.sample(available_images, sample_size)

    logger.info(f"Sampled {len(sampled_images)} images")

    # copy images to target directory
    for image_path in sampled_images:
        class_name = os.path.basename(os.path.dirname(image_path))
        class_target_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_target_dir, exist_ok=True)

        target_path = os.path.join(class_target_dir, os.path.basename(image_path))
        shutil.copy2(image_path, target_path)

    logger.info(f"Created partition in {target_dir}")
    return sampled_images

def add_to_dvc(dir_path, version):
    """
    Add directory to DVC and create a tagged version
    """
    
    # add directory to dvc
    logger.info(f"Adding {dir_path} to DVC")
    subprocess.run(["dvc", "add", dir_path], check = True)

    # add .dvc to git
    dvc_file = f"{dir_path}.dvc"
    logger.info(f"Adding {dvc_file} to git")
    subprocess.run(["git", "add", dvc_file], check = True)

    # commit the changes
    logger.info(f"Committing version {version}")
    subprocess.run(["git", "commit", "-m", f"Add dataset partition {version}"], check = True)

    # tag the version
    logger.info(f"Tagging as {version}")
    subprocess.run(["git", "tag", "-a", version,"-m",f"Dataset partition {version}"], check = True)

    logger.info(f"Successfully versioned {dir_path} as {version}")

def main():
    parser = argparse.ArgumentParser(description="Create dataset partitions and version with DVC")
    parser.add_argument("--seed", type=int, default=26, help="Random seed")
    args = parser.parse_args()

    source_dir = "data/raw"
    partition_size = 20000

    partitions = [{"dir": "data/v1", "version": "v1"},
                  {"dir": "data/v2", "version": "v2"},
                  {"dir": "data/v3", "version": "v3"}]

    already_sampled = set()

    for partition in partitions:
        logger.info(f"Creating partition {partition['version']}")

        sampled_images = create_partition(source_dir, partition['dir'], partition_size, args.seed, already_sampled)

        add_to_dvc(partition['dir'], partition['version'])

        already_sampled.update(sampled_images)

        args.seed += 1

    logger.info("All partitions created and versioned successfully")

if __name__ == "__main__":
    main()