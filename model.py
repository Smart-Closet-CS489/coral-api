# model.py

# ---- Imports ----------------------------------
from flask import Flask, jsonify, request
import numpy as np
import os
import requests
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# ---- Constants -------------------------
MODEL_DIR = "/app/models"

# ---- Private ----------------------------

# ---- Flask endpoints --------------------------------
app = Flask(__name__)

@app.route('/model/create_model', methods=['POST'])
def create_tf_model():
    data = request.json
    model_name = data.get("model_name")
    input_size = data.get("input_size")
    hidden_layers = data.get("hidden_layers", [])
    output_size = data.get("output_size")

    if not model_name or input_size is None or output_size is None:
        return jsonify({"error": "Missing required parameters"}), 400

    # In-line model creation logic
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_size,), dtype=tf.float32))

    for layer_size in hidden_layers:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

    model.add(tf.keras.layers.Dense(output_size))

    model = tfmot.quantization.keras.quantize_model(model)
    model.compile(optimizer='adam', loss='mean_squared_error')

    model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
    model.save(model_path)

    return jsonify({"message": f"Model '{model_name}' created with QAT and saved at {model_path}"}), 200


@app.route('/model/update_model', methods=['POST'])
def update_tf_model():
    data = request.get_json()

    model_name = data.get("model_name")
    new_x = np.array(data.get("features"), dtype=np.float32)
    new_y = np.array(data.get("labels"), dtype=np.float32)
    learning_rate = data.get("learning_rate", 0.001)
    epochs = data.get("epochs", 5)

    if not model_name or new_x is None or new_y is None:
        return jsonify({"error": "Missing required fields (model_name, features, labels)"}), 400

    new_x = np.clip(new_x, 0, 1)
    new_y = np.clip(new_y, 0, 1)

    # In-line model update logic
    model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
    
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error')

    model.fit(new_x, new_y, epochs=epochs, batch_size=16, verbose=1)
    model.save(model_path)

    return jsonify({"message": f"Model '{model_name}' updated with new training data!"}), 200


@app.route('/model/compile_models', methods=['POST'])
def compile_and_cocompile_models():
    data = request.json
    model_names = data.get("model_names", [])

    if not model_names:
        return jsonify({"error": "No model names provided"}), 400

    # In-line model compilation logic
    tflite_model_paths = []

    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
        
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_name}' not found"}), 404

        model = tf.keras.models.load_model(model_path)

        def representative_data_gen():
            input_shape = model.input_shape[1:]
            for _ in range(300):
                yield [np.random.uniform(0, 255, input_shape).astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_data_gen
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()

        tflite_model_path = os.path.join(MODEL_DIR, f"{model_name}.tflite")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        tflite_model_paths.append(tflite_model_path)

    # Co-compile models using external API
    files = [("models", (os.path.basename(tflite_model_path), open(tflite_model_path, 'rb'), "application/octet-stream"))
             for tflite_model_path in tflite_model_paths]

    response = requests.post(os.getenv("CORAL_EDGETPU_COMPILER_API_ADDRESS"), files=files)

    if response.status_code == 200:
        zip_filename = "compiled_models.zip"
        with open(zip_filename, 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        return jsonify({"message": "Models compiled and extracted successfully."}), 200
    else:
        return jsonify({"error": f"Failed to co-compile models. Response: {response.status_code}"}), 500
