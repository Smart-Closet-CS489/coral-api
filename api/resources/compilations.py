# compilations.py

# ---- Imports ----------------------------------
from flask import jsonify, request
import numpy as np
import os
import requests
import zipfile
import shutil
import json
import random
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# ---- Constants -------------------------
MODEL_DIR = os.getenv('DOCKER_MODEL_DIR')

# ---- Private ----------------------------

# ---- Endpoint --------------------------------
def post_compilation():
    # ---- Request body -------------------------
    data = request.json
    model_names = data.get("model_names")

    # ---- Request validation ------------------------------------
    if model_names is None or model_names == "":
        return jsonify({"error": "Model name(s) are missing or empty."}), 400

    # ---- Logic ------------------------------------
    tflite_model_paths = []

    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.tf")
        
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_name}' not found"}), 404

        model = tf.keras.models.load_model(model_path)

        def representative_data_gen():
            input_shape = model.input_shape[1:]
            for _ in range(200):
                yield [np.random.uniform(0, 1, input_shape).astype(np.float32)]  # Uniformly random values between 0 and 1

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_data_gen
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()

        tflite_model_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.tflite")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        tflite_model_paths.append(tflite_model_path)

    # Co-compile models using external API
    files = [("models", (os.path.basename(tflite_model_path), open(tflite_model_path, 'rb'), "application/octet-stream"))
             for tflite_model_path in tflite_model_paths]

    response = requests.post(os.getenv("CORAL_EDGETPU_COMPILER_API_ADDRESS"), files=files)

    if response.status_code == 200:
        zip_filename = "compiled_models.zip"
        
        # Save the zip file to disk
        with open(zip_filename, 'wb') as f:
            f.write(response.content)

        # Unzip the file to a temporary directory first to inspect the filenames
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall("temp_models_folder")

        # Loop through the files in the extracted folder
        for file_name in os.listdir("temp_models_folder"):
            # Assuming each file name starts with the model name, e.g., "example_model_edgetpu.tflite"
            # Extract the model name by splitting the file name (modify if the pattern is different)
            model_name = file_name.replace('_edgetpu.tflite', '')  # This assumes the model name is the first part of the file name
            
            # Check if the model directory exists, if not, skip or log a message
            model_folder = os.path.join(MODEL_DIR, model_name)
            if not os.path.exists(model_folder):
                return jsonify({"error": f"Model folder for '{model_name}' does not exist. Ensure the model is created first."}), 400
            
            # Move the file to its respective folder
            source_path = os.path.join("temp_models_folder", file_name)
            destination_path = os.path.join(model_folder, file_name)
            shutil.move(source_path, destination_path)

        # Clean up the temporary extraction folder
        shutil.rmtree("temp_models_folder")
        
        return jsonify({"message": "Models compiled and extracted successfully."}), 200

    else:
        return jsonify({"error": f"Failed to co-compile models. Response: {response.status_code}"}), 500

