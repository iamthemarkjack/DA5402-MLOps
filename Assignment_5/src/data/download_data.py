import os
import pickle
import urllib.request
import tarfile
import numpy as np
from PIL import Image
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/download_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("download_cifar10")

def download(data_dir):
    """
    Download the dataset and extract it
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_file = os.path.join(data_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(tar_file):
        logger.info(f"Downloading CIFAR-10 dataset from {url}")
        urllib.request.urlretrieve(url, tar_file)
        logger.info(f"Downloaded CIFAR-10 dataset to {tar_file}")

    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        logger.info(f"Extracting dataset to {extract_dir}")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        logger.info("Extraction completed")

    return extract_dir

def unpickle(file):
    """
    Load the pickled data
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def create_classes(data_dir, extract_dir):
    """
    Create folders for each class and save images
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                    'frog', 'horse', 'ship', 'truck']

    class_dirs = {}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok = True)
        class_dirs[class_name] = class_dir

    logger.info(f"Created class directories : {list(class_dirs.keys())}")

    total_images = 0
    batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    for batch_file in batch_files:
        logger.info(f"Processing {batch_file}")
        batch_path = os.path.join(extract_dir, batch_file)
        batch_data = unpickle(batch_path)

        images = batch_data[b'data']
        labels = batch_data[b'labels']

        for i, (image, label) in enumerate(zip(images, labels)):
            # reshape the image to 32 x 32 x 3
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)

            # save the image
            img = Image.fromarray(image)
            class_name = class_names[label]
            img_path = os.path.join(class_dirs[class_name], f"{batch_file}_{i}.png")

            # only save if the file doesn't exist already
            if not os.path.exists(img_path):
                img.save(img_path)
                total_images += 1

                # log the process
                if total_images % 1000 == 0:
                    logger.info(f"Saved {total_images} images as of now")

    logger.info(f"Total images saved: {total_images}")
    return class_dirs

def main():
    data_dir = "data/raw"
    logger.info("Starting the downloading and extracting process of CIFAR-10 dataset")

    extract_dir = download(data_dir)

    class_dirs = create_classes(data_dir, extract_dir)

    logger.info("CIFAR-10 Dataset preparation is complete!")

if __name__ == "__main__":
    main()