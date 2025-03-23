# training_session.py

# ---- Imports ----------------------------------
from flask import jsonify, request
import numpy as np
import os
import json
import random
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import redis
import threading

# ---- Constants -------------------------
MODEL_DIR = os.getenv('DOCKER_MODEL_DIR')
BATCH_SIZES = [2, 4, 8, 16, 32]
LEARNING_RATES = [.001, .0005, .00025, .000125, .0000625]
EPOCHS = [9, 27, 81, 243, 729]
SAMPLING_PERCENTAGES = [.01, .03, .09, .27, .81]
# for each round, the total number of datapoints needs to be 3 * the batch size in order to proceed.
# first sample the memory data points based on the current percentage for that round, and if the size does not equal 3*batch size
# then continue sampling from memory until you reach 3*batch size or you run out of memory datapoints, if you run out of memory data
# points and do not have a total dataset of size 3*batch size, stop the training at that round

# ---- Private ----------------------------
_redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

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
    if len(inputs) != len(outputs):
        return jsonify({"error": "Inputs size must match outputs size."}), 400
    if len(inputs) < 6:
        return jsonify({"error": "Must have at least 6 data pairs."}), 400

    # ---- Logic ------------------------------------
    # Transform inputs and outputs from [0, 1] to [0, 255], and keep them as float32
    if _redis_client.exists("training_session_" + model_name) == 1:
        return jsonify({"error": f"Training session is already in progress for {model_name}."}), 400
    _redis_client.set("training_session_" + model_name, 1)

    def training_session_task(model_name, inputs, outputs):
        inputs = np.clip(inputs, 0, 1)
        # outputs = np.clip(outputs, 0, 1)
        # inputs = np.clip(inputs * 255, 0, 255).astype(np.float32)
        outputs = np.clip(outputs * 255, 0, 255).astype(np.float32)

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
        print("MEMORY DATA POINTS:")
        print(len(memory_inputs))

        # Determine loss function outside the loop
        loss_functions = {
            "regression": "mean_squared_error",
            "classification": "categorical_crossentropy",
            "binary": "binary_crossentropy"
        }
        loss_function = loss_functions.get(model_data["model_type"], "mean_squared_error")

        # Compile model with current optimizer outside the loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATES[1])
        model.compile(optimizer=optimizer, loss=loss_function)  # Add 'accuracy' as a metric

        # Training loop (iterating over several rounds)
        for round_num in range(0, 5):  # 5 rounds with gradually increasing memory and batch size
            learning_rate = LEARNING_RATES[round_num]  # Track learning rate
            batch_size = BATCH_SIZES[round_num]
            epochs = EPOCHS[round_num]  # Start with 3 epochs for the first round
            memory_sample_percent = SAMPLING_PERCENTAGES[round_num]

            # If there are not enough records to satisify the batch size this round, break
            if len(inputs) + len(memory_inputs) < 3 * batch_size:
                break

            memory_sample_size = 0
            if len(inputs) + int(len(memory_inputs) * memory_sample_percent) >= 3 * batch_size:
                memory_sample_size = int(len(memory_inputs) * memory_sample_percent)
            else:
                memory_sample_size = (3 * batch_size) - len(inputs)

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
            print(f"Round {round_num} Loss={history.history['loss'][-1]:.9f}")
            print("========================================================\n")

        # Save updated model
        model.save(model_path)
        _redis_client.delete("training_session_" + model_name)

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
    
    training_thread = threading.Thread(target=training_session_task, args=(model_name, inputs, outputs))
    training_thread.start()

    return jsonify({"message": f"Training session started for model '{model_name}'!"}), 200


def get_training_session(model_name):
    training_session_ongoing = _redis_client.exists("training_session_" + model_name) == 1
    return jsonify({"training_session_active": training_session_ongoing}), 200

