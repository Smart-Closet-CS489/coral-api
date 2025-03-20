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
    memory_size = data.get("memory_size", 1000)  # Default to 1000 if not provided

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
        "memory_size": memory_size,
        "memory_inputs": [],
        "memory_outputs": []
    }

    json_file_path = os.path.join(model_folder, f"{model_name}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(model_data, json_file, indent=4)

    return jsonify({"message": f"Model '{model_name}' created and saved at {model_path}, and JSON saved at {json_file_path}"}), 200


@app.route('/model/update_model', methods=['POST'])
def update_model():
    # Constants
    INITIAL_LEARNING_RATE = 0.001
    BATCH_SIZE = 4  # Initial batch size
    MIN_BATCH_SIZE = 2
    MAX_BATCH_SIZE = 8
    EPOCHS = 80
    MEMORY_SAMPLING_PERCENT = 0.5  # Percent of stored memory to sample per training round
    LOSS_THRESHOLD = 0.01  # Change in loss required to trigger batch size change
    BATCH_ADJUST_FACTOR = 2  # Factor by which to increase/decrease batch size
    LR_DECAY_FACTOR = 0.9  # Factor to decay learning rate when loss worsens
    LR_INCREASE_FACTOR = 1.1  # Factor to increase learning rate when loss improves

    # Get request data
    data = request.get_json()
    model_name = data["model_name"]
    inputs = np.array(data["inputs"], dtype=np.float32)
    outputs = np.array(data["outputs"], dtype=np.float32)

    # Clip inputs/outputs to [0, 1] range
    inputs = np.clip(inputs, 0, 1)
    outputs = np.clip(outputs, 0, 1)

    # Load model
    model_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.tf")
    model = tf.keras.models.load_model(model_path)

    # Load memory from JSON
    json_file_path = os.path.join(MODEL_DIR, f"{model_name}/{model_name}.json")
    with open(json_file_path, 'r') as f:
        model_data = json.load(f)

    memory_inputs = model_data["memory_inputs"]
    memory_outputs = model_data["memory_outputs"]
    memory_size = model_data["memory_size"]

    # Sample memory (randomly select a percentage of stored memory)
    memory_sample_size = int(len(memory_inputs) * MEMORY_SAMPLING_PERCENT)
    sampled_indices = random.sample(range(len(memory_inputs)), memory_sample_size) if memory_inputs else []
    sampled_inputs = np.array([memory_inputs[i] for i in sampled_indices])
    sampled_outputs = np.array([memory_outputs[i] for i in sampled_indices])

    # Combine sampled memory with new inputs
    extended_inputs = np.concatenate((inputs, sampled_inputs)) if sampled_inputs.size else inputs
    extended_outputs = np.concatenate((outputs, sampled_outputs)) if sampled_outputs.size else outputs

    # Determine loss function
    loss_functions = {
        "regression": "mean_squared_error",
        "classification": "categorical_crossentropy",
        "binary": "binary_crossentropy"
    }
    loss_function = loss_functions.get(model_data["model_type"], "mean_squared_error")

    # Compile model with initial optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_function)

    prev_loss = float("inf")  # Track previous epoch loss
    learning_rate = INITIAL_LEARNING_RATE  # Track learning rate

    # Training loop
    for epoch in range(EPOCHS):
        # Update learning rate dynamically before training
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)

        # Train model
        history = model.fit(extended_inputs, extended_outputs, epochs=1, batch_size=BATCH_SIZE, verbose=0)
        avg_loss = history.history["loss"][0]

        # Print diagnostics
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.6f}, Batch Size={BATCH_SIZE}, Learning Rate={learning_rate:.6f}")

        # Adjust batch size
        if abs(prev_loss - avg_loss) < LOSS_THRESHOLD:
            BATCH_SIZE = min(BATCH_SIZE * BATCH_ADJUST_FACTOR, MAX_BATCH_SIZE)  # Increase batch size
        elif avg_loss > prev_loss:
            BATCH_SIZE = max(MIN_BATCH_SIZE, BATCH_SIZE // BATCH_ADJUST_FACTOR)  # Decrease batch size

        # Adjust learning rate
        if avg_loss < prev_loss:
            learning_rate *= LR_INCREASE_FACTOR  # Slightly increase learning rate if loss improves
        else:
            learning_rate *= LR_DECAY_FACTOR  # Decay learning rate if loss worsens

        prev_loss = avg_loss  # Update previous loss

    # Save updated model
    model.save(model_path)

    # Update memory (prepend new inputs & outputs)
    memory_inputs = inputs.tolist() + memory_inputs
    memory_outputs = outputs.tolist() + memory_outputs

    # Trim memory if it exceeds the buffer size
    if len(memory_inputs) > memory_size:
        memory_inputs = memory_inputs[:memory_size]
        memory_outputs = memory_outputs[:memory_size]

    # Save updated memory to JSON
    model_data["memory_inputs"] = memory_inputs
    model_data["memory_outputs"] = memory_outputs
    with open(json_file_path, 'w') as f:
        json.dump(model_data, f, indent=4)

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

