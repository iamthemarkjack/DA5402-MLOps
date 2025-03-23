# CIFAR-10 DVC Pipeline

This repository provides a structured pipeline for downloading, organizing, and processing the CIFAR-10 dataset and building, training and experimenting a CNN multi-classifier using DVC (Data Version Control). Follow the steps below to initialize the project and run experiments.

## Setup

### 1. Initialize the Project
To initialize the project, first, run the following commands:
```sh
git init
dvc init
```
- `git init` initializes a Git repository.
- `dvc init` initializes a DVC repository for data versioning and pipeline management.

### 2. Download and Organize the CIFAR-10 Dataset
Run the script to download and organize the CIFAR-10 dataset:
```sh
python src/data/download_dataset.py
```
This script will:
- Download the CIFAR-10 dataset.
- Organize it into a structured format.

### 3. Create and Version the Dataset Partitions
Once the dataset is downloaded, partition it using:
```sh
python src/data/create_partitions.py
```
This step:
- Splits the dataset into three partitions _v1_, _v2_ and _v3_ and pushes them into DVC.

### 4. Run the DVC Pipeline with Different Configurations
Execute the experiments using the predefined DVC pipeline:
```sh
python run_experiments.py
```
This script will:
- Run model training with different configurations as given in the assignment question.
- Log and track results for reproducibility.

## Notes
- Ensure all dependencies are installed before running the scripts. You can install them using:
  ```sh
  pip install -r requirements.txt
  ```
- Use `dvc repro` to rerun the pipeline with tracked dependencies.
- Use `dvc metrics show` to view the performance metrics of different experiments.
