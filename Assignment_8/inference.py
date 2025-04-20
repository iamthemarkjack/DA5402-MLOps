import os
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow
import argparse
import logging
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, img_width=128, img_height=32):
        self.img_width = img_width
        self.img_height = img_height
        
        self.char_to_num = layers.StringLookup(
            vocabulary=list(string.ascii_lowercase + string.digits + " "), mask_token=None
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def process_image(image_data, config):
    """Process image for model input."""
    try:
        # convert to grayscale if needed
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            image_data = tf.image.rgb_to_grayscale(image_data)
        elif len(image_data.shape) == 2:
            image_data = tf.expand_dims(image_data, axis=-1)
            
        # convert to float32 in [0, 1]
        image_data = tf.image.convert_image_dtype(image_data, tf.float32)
        
        # resize image
        image_data = distortion_free_resize(image_data, (config.img_width, config.img_height))
        
        # ensure correct shape for model
        if len(image_data.shape) < 3:
            image_data = tf.expand_dims(image_data, axis=-1)
            
        # add batch dimension
        image_data = tf.expand_dims(image_data, axis=0)
        
        return image_data
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def decode_predictions(pred, config):
    """Decode model predictions to text."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # use greedy search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    # convert to text
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        res = tf.squeeze(res)
        res = tf.strings.reduce_join(config.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
        
    return output_text[0] if output_text else ""


def load_model(model_uri):
    """Load model from MLflow model registry."""
    try:
        logger.info(f"Loading model from {model_uri}")
        model = mlflow.tensorflow.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


# Flask application for API
app = Flask(__name__)
model = None
config = None


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to recognize handwritten text from an image."""
    if request.method == 'POST':
        try:
            # get image from request
            if 'image' not in request.files and 'image' not in request.form:
                return jsonify({'error': 'No image provided'}), 400
            
            if 'image' in request.files:
                # process image file
                image_file = request.files['image']
                img = Image.open(image_file)
                img = img.convert('L')
                img_array = np.array(img)
            else:
                # process base64 encoded image
                image_b64 = request.form['image']
                img_data = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_data))
                img = img.convert('L')
                img_array = np.array(img)
            
            # process image
            processed_img = process_image(img_array, config)
            
            #  ake prediction
            predictions = model.predict(processed_img)
            
            # decode prediction
            recognized_text = decode_predictions(predictions, config)
            
            return jsonify({
                'recognized_text': recognized_text
            }), 200
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Handwriting recognition API server")
    parser.add_argument("--model_uri", type=str, required=True, 
                        help="MLflow model URI (e.g., 'runs:/run_id/handwriting_recognition_model' or 'models:/HandwritingRecognitionModel/1')")
    parser.add_argument("--port", type=int, default=5000, help="Port for the API server")
    parser.add_argument("--img_width", type=int, default=128, help="Image width")
    parser.add_argument("--img_height", type=int, default=32, help="Image height")
    
    return parser.parse_args()


def main():
    """Main function to start the API server."""
    global model, config
    
    args = parse_args()
    
    config = Config(img_width=args.img_width, img_height=args.img_height)