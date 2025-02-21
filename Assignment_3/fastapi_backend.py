import os
import numpy as np
from PIL import Image
import io
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from urllib.parse import urlparse

from utils import *
from dense_neural_class import *

# Function to load the model with absolute path
def load_model(filename):
    # Gets the current directory where the script is being executed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Constructs the full path of the .pkl file
    filepath = os.path.join(current_dir, filename + '.pkl')
    
    with open(filepath, 'rb') as file:
        model_loaded = pickle.load(file)
    
    return model_loaded

app = FastAPI()
model = load_model('model')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")
        image = np.array(image).reshape(1,-1)/255.0
        prediction = model.predict(image)[0]

        return JSONResponse({"prediction" : str(prediction)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict", help="API Endpoint URL")
    url = parser.parse_args().url

    try:
        parsed_url = urlparse(url)
    except Exception as e:
        print(f"Error parsing URL: {e}")
        parsed_url = None
    try:
        host = parsed_url.hostname if parsed_url else None
    except Exception as e:
        print(f"Error extracting hostname: {e}")
        host = None
    try:
        port = parsed_url.port if parsed_url else None
    except Exception as e:
        print(f"Error extracting port: {e}")
        port = None

    if host and port:
        uvicorn.run(app, host=host, port=port)
