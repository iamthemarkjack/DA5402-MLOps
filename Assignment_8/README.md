# Handwriting Recognition with MLflow

This project implements a handwriting recognition system using TensorFlow/Keras and MLflow for experiment tracking and model management. The model is trained to recognize handwritten text from images.

## Project Structure

```
project_root/
├── handwriting_recognition.py   # Main model training script with MLflow tracking
├── inference.py                 # Inference script for REST API
├── infer_command.py             # Command-line inference script
├── MLproject                    # MLflow project definition
├── conda.yaml                   # Environment dependencies
└── README.md                    # Project documentation
```

## Setup

1. Clone this repository
2. Install MLflow: `pip install mlflow`
3. Ensure conda is installed (for environment management)

## Usage

### Training

Train the model with MLflow tracking:

```bash
# Basic training with default parameters
mlflow run . -e train

# Training with custom parameters
mlflow run . -e train -P batch_size=32 -P epochs=50 -P train_ratio=0.75 -P val_ratio=0.15
```

The training script will:
- Load and preprocess the IAM handwriting dataset
- Train a CNN-RNN model with CTC loss
- Track experiments with MLflow (metrics, parameters, artifacts)
- Register the model in the MLflow registry
- Generate visualizations of training progress

### Serving as a REST API

Serve the trained model as a REST API:

```bash
# Using a specific run ID
mlflow run . -e serve -P model_uri="runs:/<run_id>/handwriting_recognition_model"

# Using a registered model version
mlflow run . -e serve -P model_uri="models:/HandwritingRecognitionModel/1"
```

The API will be available at `http://localhost:5000/predict` and accepts POST requests with an image file or base64-encoded image.

### Command-line Inference

Run inference on a single image:

```bash
# Using a specific run ID
mlflow run . -e infer -P model_uri="runs:/<run_id>/handwriting_recognition_model" -P image_path="path/to/image.png"

# Using a registered model version
mlflow run . -e infer -P model_uri="models:/HandwritingRecognitionModel/1" -P image_path="path/to/image.png"
```

## MLflow Tracking

The training script logs the following to MLflow:

### Parameters
- Batch size
- Image dimensions
- Train/val/test split ratios
- Maximum sequence length
- Number of epochs

### Metrics
- Training loss
- Validation loss
- Test loss
- Edit distance (CER - Character Error Rate)

### Artifacts
- Trained model
- Loss plots
- Edit distance plots
- Sample test images

## API Usage with curl

```bash
# Using an image file
curl -X POST -F "image=@path/to/image.png" http://localhost:5000/predict

# Using base64 encoded image
curl -X POST -F "image=$(base64 -w 0 path/to/image.png)" http://localhost:5000/predict
```