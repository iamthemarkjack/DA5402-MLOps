import os
import argparse
import logging
import subprocess
import glob
import shutil
import pandas as pd
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pull_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pull_data")

def checkout_version(version, seed):
    """
    Checkout a specific version from DVC
    If version contains '+', it's a combination of multiple version
    """

    versions = version.split('+')
    output_dir = "data/prepared"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    for v in versions:
        logger.info(f"Checking out dataset version: {v}")

        subprocess.run(["git", "checkout", "-f", v], check = True)
        subprocess.run(["dvc", "checkout", f"data/{v}.dvc"], check = True)

        # copy files to prepared directory
        source_dir = f"data/{v}"
        if os.path.exists(source_dir):
            class_dirs = next(os.walk(source_dir))[1]
            for class_dir in class_dirs:
                source_class_dir = os.path.join(source_dir, class_dir)
                target_class_dir = os.path.join(output_dir, class_dir)

                os.makedirs(target_class_dir, exist_ok = True)

                for img_file in glob.glob(os.path.join(source_class_dir, "*.png")):
                    shutil.copy2(img_file, target_class_dir)
        else:
            logger.error(f"Source directory {source_dir} does not exist")

    subprocess.run(["git", "checkout", "main"], check = True)

    create_dataframe(output_dir, seed)

def create_dataframe(image_dir, seed):
    """
    Create a DataFrame from the image directory
    """

    np.random.seed(seed)

    data = []
    class_names = sorted(next(os.walk(image_dir))[1])

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        for img_path in glob.glob(os.path.join(class_dir, "*.png")):
            img = Image.open(img_path)
            img_array = np.array(img)

            data.append({
                'image_path': img_path,
                'class_name': class_name,
                'class_idx': class_idx,
                'image_data': img_array
            })

    df = pd.DataFrame(data)

    # shuffle the DataFrame
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Created DataFrame with {len(df)} images")

    # save the DataFrame
    df.to_pickle("data/prepared/dataset.pkl")
    logger.info("Saved DataFrame to data/prepared/dataset.pkl")

    # save the summary of class distributions
    class_distribution = df['class_name'].value_counts().reset_index()
    class_distribution.columns = ['class_name', 'count']
    class_distribution.to_csv("data/prepared/class_distribution.csv", index=False)
    logger.info("Saved class distribution to data/prepared/class_distribution.csv")

def main():
    parser = argparse.ArgumentParser(description="Pull a specific dataset version from DVC")
    parser.add_argument("--version", type=str, required=True, help="Dataset version to pull")
    parser.add_argument("--seed", type=int, default=26, help="Random seed")
    args = parser.parse_args()
    
    logger.info(f"Pulling dataset version: {args.version} with seed: {args.seed}")
    checkout_version(args.version, args.seed)
    logger.info("Dataset pull completed")

if __name__ == "__main__":
    main()