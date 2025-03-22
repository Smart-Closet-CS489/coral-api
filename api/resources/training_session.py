# training_session.py

# ---- Imports ----------------------------------
from flask import jsonify, request
import numpy as np
import os
import json
import random
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# ---- Constants -------------------------
MODEL_DIR = os.getenv('DOCKER_MODEL_DIR')
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_RATIO = .6
INITIAL_BATCH_SIZE = 2  # Initial batch size
BATCH_SIZE_RATIO = 2
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 32
EPOCHS = 80
MEMORY_SAMPLING_INITAL = 0.1  # Percent of stored memory to sample per training round
MEMORY_SAMPLING_INCREMEMT = 0.15
EPOCH_INCREMENT = 130  # Increase epochs per round
INITAL_EPOCHS = 30

# ---- Private ----------------------------

# ---- Endpoints --------------------------------
def post_training_session(model_name):
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
    # Transform inputs and outputs from [0, 1] to [0, 255], and keep them as float32
    inputs = np.clip(inputs, 0, 1)
    outputs = np.clip(outputs, 0, 1)
    # inputs = np.clip(inputs * 255, 0, 255).astype(np.float32)
    outputs = np.clip(outputs * 254, 0, 254).astype(np.float32)

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

    # Determine loss function outside the loop
    loss_functions = {
        "regression": "mean_squared_error",
        "classification": "categorical_crossentropy",
        "binary": "binary_crossentropy"
    }
    loss_function = loss_functions.get(model_data["model_type"], "mean_squared_error")

    # Initialize training variables
    learning_rate = INITIAL_LEARNING_RATE  # Track learning rate
    batch_size = INITIAL_BATCH_SIZE
    epochs = INITAL_EPOCHS  # Start with 3 epochs for the first round
    memory_sample_percent = MEMORY_SAMPLING_INITAL

    # Compile model with current optimizer outside the loop
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])  # Add 'accuracy' as a metric

    # Training loop (iterating over several rounds)
    for round_num in range(1, 6):  # 5 rounds with gradually increasing memory and batch size
        # Dynamically sample memory based on the current round
        memory_sample_size = int(len(memory_inputs) * memory_sample_percent)  # Sample 10%, 20%, ..., 50% of memory
        sampled_indices = random.sample(range(len(memory_inputs)), memory_sample_size) if memory_inputs else []
        sampled_inputs = np.array([memory_inputs[i] for i in sampled_indices])
        sampled_outputs = np.array([memory_outputs[i] for i in sampled_indices])

        # Combine sampled memory with new inputs
        extended_inputs = np.concatenate((inputs, sampled_inputs)) if sampled_inputs.size else inputs
        extended_outputs = np.concatenate((outputs, sampled_outputs)) if sampled_outputs.size else outputs

        # Update learning rate dynamically before training
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)

        # Train the model for the current round
        print("\n========================================================")
        print(f"Starting Round {round_num} -- [batch size: {batch_size}]  [learning rate: {learning_rate:.6f}]  [epochs: {epochs}]  [dataset size: {len(extended_inputs)}]")
        history = model.fit(extended_inputs, extended_outputs, epochs=epochs, batch_size=batch_size, verbose=0)

        # Print diagnostics: Loss and accuracy
        print(f"Round {round_num} Loss={history.history['loss'][-1]:.9f}, Accuracy={history.history['accuracy'][-1]:.6f}")
        print("========================================================\n")

        # Adjust batch size and learning rate for next round
        batch_size = min(batch_size * BATCH_SIZE_RATIO, MAX_BATCH_SIZE)  # Double the batch size, but cap it at MAX_BATCH_SIZE
        learning_rate *= LEARNING_RATE_RATIO  # Decrease learning rate by 0.0004 each round
        memory_sample_percent += MEMORY_SAMPLING_INCREMEMT
        epochs += EPOCH_INCREMENT

    # Save updated model
    model.save(model_path)

    predictions = model.predict(inputs)
    for i in range(min(5, len(inputs))):  # Print first 5 samples or as many as you have
        print(f"Test sample {i}:")
        print("Input:", inputs[i])  # Each input could be a variable-length array
        print("True output:", outputs[i])  # Expected output
        print("Predicted output:", predictions[i])  # Model's predicted output
        print("---------------------------")

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
