# models.py

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

# ---- Endpoints --------------------------------
def post_model():
    # ---- Request body -------------------------
    data = request.json
    model_name = data.get("model_name")
    input_size = data.get("input_size")
    hidden_sizes = data.get("hidden_sizes")
    output_size = data.get("output_size")
    model_type = data.get("model_type")
    memory_size = data.get("memory_size")

    # ---- Request validation ------------------------------------
    if model_name is None or model_name == "":
        return jsonify({"error": "Model name is missing or empty."}), 400
    if input_size is None or not isinstance(input_size, int):
        errors.append("'input_size' is required and must be an integer.")
    if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
        errors.append("'hidden_sizes' is required and must be a non-empty list.")
    if output_size is None or not isinstance(output_size, int):
        errors.append("'output_size' is required and must be an integer.")
    if model_type is None or model_type not in ["regression", "classification", "binary"]:
        errors.append("'model_type' is required and must be one of ['regression', 'classification', 'binary'].")
    if memory_size is None or not isinstance(memory_size, int) or memory_size <= 0:
        errors.append("'memory_size' is required and must be a positive integer.")

    # ---- Logic -----------------------------------------
    model_folder = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(model_folder):
        return jsonify({"error": f"Model '{model_name}' already exists, if you wish to overwrite it you must first delete it."}), 400

    activation_function = ""
    if model_type == "classification":
        activation_function = "softmax"
    elif model_type == "binary":
        activation_function = "sigmoid"
    else:
        model_type = "regression"
        activation_function = "linear"

    if not model_name or input_size is None or output_size is None:
        return jsonify({"error": "Missing required parameters"}), 400

    # In-line model creation logic
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_size,), dtype=tf.float32))

    for layer_size in hidden_sizes:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

    model.add(tf.keras.layers.Dense(output_size, activation=activation_function))

    model = tfmot.quantization.keras.quantize_model(model)

    # Save the model
    model_folder = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_folder, exist_ok=True)

    # Save the model file
    model_path = os.path.join(model_folder, f"{model_name}.tf")
    model.save(model_path)

    print(model.summary())

    # Create and save the JSON file
    model_data = {
        "model_name": model_name,
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "model_type": model_type,
        "memory_size": memory_size,
        "memory_inputs": [],
        "memory_outputs": []
    }

    json_file_path = os.path.join(model_folder, f"{model_name}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(model_data, json_file, indent=4)

    return jsonify({"message": f"Model '{model_name}' created and saved at {model_path}, and JSON saved at {json_file_path}"}), 200


def get_model(model_name):
    # ---- Request validation ------------------------------------
    if model_name is None or model_name == "":
        return jsonify({"error": "Model name is missing or empty."}), 400

    # ---- Logic -----------------------------------------
    model_folder = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_folder):
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    json_file_path = os.path.join(model_folder, f"{model_name}.json")

    if not os.path.exists(json_file_path):
        return jsonify({"error": f"JSON file for model '{model_name}' not found."}), 404

    try:
        with open(json_file_path, 'r') as json_file:
            model_data = json.load(json_file)

        # Remove memory inputs and outputs
        model_data.pop("memory_inputs", None)
        model_data.pop("memory_outputs", None)

        return jsonify(model_data), 200
    except Exception as e:
        return jsonify({"error": f"Error reading model data: {str(e)}"}), 500


def delete_model(model_name):
    # ---- Request validation -----------------------------
    if model_name is None or model_name == "":
        return jsonify({"error": "Model name is missing or empty."}), 400

    # ---- Logic -----------------------------------------
    model_folder = os.path.join(MODEL_DIR, model_name)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        return jsonify({"error": f"Model '{model_name}' does not exist."}), 404

    try:
        # Remove the entire model directory
        shutil.rmtree(model_folder)
        return jsonify({"message": f"Model '{model_name}' and its directory have been deleted."}), 200
    except Exception as e:
        return jsonify({"error": f"Error deleting the model: {str(e)}"}), 500