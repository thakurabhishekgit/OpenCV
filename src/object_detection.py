import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import cv2
from keras.models import load_model

import h5py

file_path = 'models/mobile_classification_model.h5'
try:
    with h5py.File(file_path, 'r') as f:
        print("File loaded successfully")
except Exception as e:
    print(f"Error opening HDF5 file: {e}")
file_path = 'models/mobile_classification_model.h5'
print(f"File exists: {os.path.exists(file_path)}")
print(f"File readable: {os.access(file_path, os.R_OK)}")


model_path = 'models/mobile_classification_model.h5'
if not os.path.isfile(model_path):
    print(f"Model file not found at {model_path}")
else:
    print(f"Model file found at {model_path}")

model = None  # Initialize model variable

model = None

def load_my_model():
    global model
    model_path = 'models/mobile_classification_model.h5'
    
    if os.path.exists(model_path):
        print(f"Attempting to load model from: {model_path}")
        try:
            model = load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {model_path}")



# Call this function at the start of your application to load the model
load_my_model()

def detect_objects(image):
    global model  # Ensure to use the global model variable
    if model is None:
        raise RuntimeError("Model is not loaded. Please call `load_my_model()` first.")
    
    # Resize and preprocess the image to match the input shape of the model
    resized_image = cv2.resize(image, (224, 224))
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make predictions using the model
    predictions = model.predict(img_array)
    
    # Decode the predictions into human-readable labels
    labels = decode_predictions(predictions)
    
    # Extract the most likely object and its confidence score
    top_prediction = labels[0][0]
    object_name, confidence = top_prediction[1], top_prediction[2]
    
    return object_name, confidence
