# inference.py

# ---- Imports ----------------------------------
import time
from flask import Flask, jsonify, request
import numpy as np
import os
import pycoral.utils.edgetpu as edgetpu

# ---- Constants -------------------------
MODEL_DIR = "/app/models"

# ---- Private ----------------------------
CACHE_TIMEOUT = 10
cached_models = {}

def load_model(model_name):
    model_path = f"/app/models/{model_name}/{model_name}_edgetpu.tflite"
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

# ---- Flask endpoints --------------------------------
app = Flask(__name__)

@app.route('/inference/run_inference', methods=['POST'])
def run_inference_on_coral():
    data = request.json
    model_name = data.get("model_name")
    inputs = np.array(data.get("features"), dtype=np.float32)

    # Scale inputs to 0-255 (from 0-1 range)
    inputs = np.clip(inputs * 255, 0, 255).astype(np.uint8)

    input_shape = inputs.shape
    if len(input_shape) == 1:
        inputs = np.expand_dims(inputs, axis=0)

    # Check if model is cached and if it's still valid (based on timeout)
    current_time = time.time()
    if model_name in cached_models:
        model_data = cached_models[model_name]
        if current_time - model_data['load_time'] > CACHE_TIMEOUT:
            # Cache expired, reload the model
            print(f"Cache expired for {model_name}. Reloading model.")
            interpreter = load_model(model_name)
            cached_models[model_name] = {'interpreter': interpreter, 'load_time': current_time}
        else:
            # Use cached model
            interpreter = model_data['interpreter']
    else:
        # Model not cached or cache expired, load the model
        print(f"Loading model {model_name} for the first time.")
        interpreter = load_model(model_name)
        cached_models[model_name] = {'interpreter': interpreter, 'load_time': current_time}

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    results = []
    for i in range(inputs.shape[0]):
        input_data = inputs[i]
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        output_data = np.clip(output_data, 0, 255).astype(np.float32) / 255.0

        results.append(output_data.flatten().tolist())

    return jsonify({"output": results}), 200
