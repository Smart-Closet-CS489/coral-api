# model.py

# ---- Imports ----------------------------------
from flask import Flask, jsonify, request
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
MODEL_DIR = "/app/models"

# ---- Private ----------------------------

# ---- Flask endpoints --------------------------------
app = Flask(__name__)

@app.route('/model/create_model', methods=['POST'])
def create_model():
    data = request.json
    model_name = data.get("model_name")
    input_size = data.get("input_size")
    hidden_sizes = data.get("hidden_sizes", [])
    output_size = data.get("output_size")
    model_type = data.get("model_type")
    history_buffer_size = data.get("history_buffer_size", 100)  # Default to 100 if not provided

    activation_function = ""
    if model_type == "regression":
        activation_function = "linear"
    elif model_type == "classification":
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

    # Create and save the JSON file
    model_data = {
        "model_name": model_name,
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "model_type": model_type,
        "history_buffer_size": history_buffer_size,
        "history_inputs": [],
        "history_outputs": []
    }

    json_file_path = os.path.join(model_folder, f"{model_name}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(model_data, json_file, indent=4)

    return jsonify({"message": f"Model '{model_name}' created and saved at {model_path}, and JSON saved at {json_file_path}"}), 200


@app.route('/model/update_model', methods=['POST'])
def update_model():
    data = request.get_json()
    model_name = data.get("model_name")
    inputs = np.array(data.get("inputs"), dtype=np.float32)
    outputs = np.array(data.get("outputs"), dtype=np.float32)
    learning_rate = data.get("learning_rate", 0.001)
    epochs = data.get("epochs", 5)
    batch_size = data.get("batch_size", 16)
    history_percent = data.get("history_percent", None)  # The percentage of historical data to sample (None if not provided)

    if not model_name or inputs is None or outputs is None:
        return jsonify({"error": "Missing required fields (model_name, inputs, outputs)"}), 400

    # Clip inputs and outputs to [0, 1] range
    inputs = np.clip(inputs, 0, 1)
    outputs = np.clip(outputs, 0, 1)

    # Load the model
    model_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.tf")
    
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    model = tf.keras.models.load_model(model_path)
    
    # Read the existing model's JSON file to get history data
    json_file_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.json")
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            model_data = json.load(f)
        
        history_inputs = model_data.get("history_inputs", [])
        history_outputs = model_data.get("history_outputs", [])
        history_buffer_size = model_data.get("history_buffer_size", 100)  # Get history buffer size from the JSON file
    else:
        history_inputs = []
        history_outputs = []
        history_buffer_size = 100  # Default buffer size if not found in JSON

    # Step 1: Sample history based on history_percent if history_inputs is not empty
    if history_percent and len(history_inputs) > 0:
        history_sample_size = int(len(history_inputs) * history_percent)
        sampled_indices = random.sample(range(len(history_inputs)), history_sample_size)
        
        sampled_inputs = [history_inputs[i] for i in sampled_indices]
        sampled_outputs = [history_outputs[i] for i in sampled_indices]
    else:
        sampled_inputs = []
        sampled_outputs = []

    # Step 2: Add new inputs and outputs to the beginning of the history, if history exists
    if len(history_inputs) > 0:
        history_inputs = inputs.tolist() + history_inputs
        history_outputs = outputs.tolist() + history_outputs
    else:
        # If no history exists, just use the current inputs and outputs
        history_inputs = inputs.tolist()
        history_outputs = outputs.tolist()

    # Step 3: Trim the history if it exceeds the history buffer size
    if len(history_inputs) > history_buffer_size:
        excess_size = len(history_inputs) - history_buffer_size
        history_inputs = history_inputs[:history_buffer_size]
        history_outputs = history_outputs[:history_buffer_size]

    # Step 4: Add the sampled inputs and outputs to the new inputs/outputs only if sampled data exists
    if len(sampled_inputs) > 0:
        extended_inputs = np.concatenate((inputs, np.array(sampled_inputs)))
        extended_outputs = np.concatenate((outputs, np.array(sampled_outputs)))
    else:
        extended_inputs = inputs
        extended_outputs = outputs

    # Update the model's history in the JSON file
    model_data["history_inputs"] = history_inputs
    model_data["history_outputs"] = history_outputs

    with open(json_file_path, 'w') as f:
        json.dump(model_data, f, indent=4)

    # Compile the model with the correct loss function
    if model_data["model_type"] == "regression":
        loss_function = 'mean_squared_error'
    elif model_data["model_type"] == "classification":
        loss_function = 'categorical_crossentropy'
    elif model_data["model_type"] == "binary":
        loss_function = 'binary_crossentropy'
    else:
        loss_function = 'mean_squared_error'  # Default to regression

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function)

    # Train the model with the extended inputs and outputs (current data + sampled history)
    model.fit(extended_inputs, extended_outputs, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_path)

    return jsonify({"message": f"Model '{model_name}' updated with new training data!"}), 200
    

@app.route('/model/compile_models', methods=['POST'])
def compile_models():
    data = request.json
    model_names = data.get("model_names", [])

    if not model_names:
        return jsonify({"error": "No model names provided"}), 400

    # In-line model compilation logic
    tflite_model_paths = []

    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.tf")
        
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_name}' not found"}), 404

        model = tf.keras.models.load_model(model_path)

        def representative_data_gen():
            input_shape = model.input_shape[1:]
            for _ in range(100):
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

