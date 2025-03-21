# train.py

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
INITIAL_LEARNING_RATE = 0.001
INITIAL_BATCH_SIZE = 4  # Initial batch size
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 8
EPOCHS = 80
MEMORY_SAMPLING_PERCENT = 0.5  # Percent of stored memory to sample per training round
LOSS_THRESHOLD = 0.01  # Change in loss required to trigger batch size change
BATCH_ADJUST_FACTOR = 2  # Factor by which to increase/decrease batch size
LR_DECAY_FACTOR = 0.9  # Factor to decay learning rate when loss worsens
LR_INCREASE_FACTOR = 1.1  # Factor to increase learning rate when loss improves

# ---- Private ----------------------------


# ---- Endpoint --------------------------------
def train():
    # ---- Query paramters -----------------------------
    model_name = request.args.get('model_name')
    
    # ---- Request body -------------------------
    data = request.get_json()
    inputs = np.array(data["inputs"], dtype=np.float32)
    outputs = np.array(data["outputs"], dtype=np.float32)

    # ---- Request validation ------------------------------------
    if model_name is None or model_name == "":
        return jsonify({"error": "Model name is missing or empty."}), 400
    if inputs is None or len(inputs) == 0:
        return jsonify({"error": "Inputs are missing or empty."}), 400
    if outputs is None or len(outputs) == 0:
        return jsonify({"error": "Outputs are missing or empty."}), 400

    # ---- Logic ------------------------------------
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
    BATCH_SIZE = INITIAL_BATCH_SIZE

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
