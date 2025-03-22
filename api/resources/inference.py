# inference.py

# ---- Imports ----------------------------------
import time
from flask import jsonify, request
import numpy as np
import os
import pycoral.utils.edgetpu as edgetpu

# ---- Constants -------------------------
MODEL_DIR = os.getenv('DOCKER_MODEL_DIR')
CACHE_TIMEOUT = 10

# ---- Private ----------------------------
_cached_models = {}

def _load_model(model_name):
    model_path = f"/app/models/{model_name}/{model_name}_edgetpu.tflite"
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

# ---- Endpoints ---------------------
def post_inference(model_name):
    # ---- Request body -------------------------
    data = request.json
    inputs = np.array(data.get("inputs"), dtype=np.float32)

    # ---- Request validation ------------------------------------
    if model_name is None or model_name == "":
        return jsonify({"error": "Model name is missing or empty."}), 400
    if inputs is None or len(inputs) == 0:
        return jsonify({"error": "Inputs are missing or empty."}), 400

    # ---- Logic ------------------------------------
    inputs = np.clip(inputs * 255, 0, 255).astype(np.uint8)

    input_shape = inputs.shape
    if len(input_shape) == 1:
        inputs = np.expand_dims(inputs, axis=0)

    current_time = time.time()
    if model_name in _cached_models:
        model_data = _cached_models[model_name]
        if current_time - model_data['load_time'] > CACHE_TIMEOUT:
            print(f"Cache expired for {model_name}. Reloading model.")
            interpreter = _load_model(model_name)
            _cached_models[model_name] = {'interpreter': interpreter, 'load_time': current_time}
        else:
            interpreter = model_data['interpreter']
    else:
        print(f"Loading model {model_name} for the first time.")
        interpreter = _load_model(model_name)
        _cached_models[model_name] = {'interpreter': interpreter, 'load_time': current_time}

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

    return jsonify({"outputs": results}), 200
