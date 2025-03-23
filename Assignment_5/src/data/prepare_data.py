import os
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/prepare_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prepare_data")

def load_params():
    """
    Load parameters from params.yaml
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def prepare_data(seed):
    """
    Split data into training, validation, and test sets
    """
    #load parameters
    params = load_params()
    train_ratio = params['data']['train_ratio']
    val_ratio = params['data']['val_ratio']
    test_ratio = params['data']['test_ratio']
    
    # verify that ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        logger.warning(f"Split ratios do not sum to 1: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
        
        # normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        logger.info(f"Normalized ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # load the prepared dataset
    df = pd.read_pickle("data/prepared/dataset.pkl")
    logger.info(f"Loaded dataset with {len(df)} samples")
    
    # calculate the effective test ratio for the first split
    effective_test_ratio = test_ratio / (1 - train_ratio)
    
    # first split: train and (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df['class_idx']
    )
    
    # second split: val and test from the remaining data
    val_df, test_df = train_test_split(
        temp_df, test_size=effective_test_ratio, random_state=seed, stratify=temp_df['class_idx']
    )
    
    logger.info(f"Split dataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    train_df.to_pickle("data/train/train.pkl")
    val_df.to_pickle("data/val/val.pkl")
    test_df.to_pickle("data/test/test.pkl")
    
    # save summary statistics
    save_summary("data/train", train_df)
    save_summary("data/val", val_df)
    save_summary("data/test", test_df)
    
    logger.info("Data preparation completed")

def save_summary(directory, df):
    """
    Save summary statistics for a dataset
    """
    # class distribution
    class_distribution = df['class_name'].value_counts().reset_index()
    class_distribution.columns = ['class_name', 'count']
    class_distribution.to_csv(f"{directory}/class_distribution.csv", index=False)
    
    # basic statistics
    stats = {
        'total_samples': len(df),
        'num_classes': df['class_idx'].nunique(),
        'class_names': df['class_name'].unique().tolist(),
        'class_distribution': df['class_name'].value_counts().to_dict()
    }
    
    # save as YAML
    with open(f"{directory}/stats.yaml", "w") as f:
        yaml.dump(stats, f)

def main():
    parser = argparse.ArgumentParser(description="Prepare data by splitting into train, validation, and test sets")
    parser.add_argument("--seed", type=int, default=26, help="Random seed")
    args = parser.parse_args()
    
    logger.info(f"Preparing data with seed: {args.seed}")
    prepare_data(args.seed)

if __name__ == "__main__":
    main()