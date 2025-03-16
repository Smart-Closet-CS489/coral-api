from flask import Flask, jsonify, request
import numpy as np
import os
import multiprocessing
import requests
import zipfile
import tensorflow_model_optimization as tfmot

app = Flask(__name__)

CORAL_EDGETPU_COMPILER_API_ADDRESS = os.getenv("CORAL_EDGETPU_COMPILER_API_ADDRESS")
MODEL_DIR = "/app/models"

@app.route('/create_tf_model', methods=['POST'])
def create_tf_model():
    data = request.json
    model_name = data.get("model_name")
    input_size = data.get("input_size")
    hidden_layers = data.get("hidden_layers", [])
    output_size = data.get("output_size")

    if not model_name or input_size is None or output_size is None:
        return jsonify({"error": "Missing required parameters"}), 400

    # Create a multiprocessing queue to handle results from the subprocess
    output_queue = multiprocessing.Queue()

    def create_model_process(model_name, input_size, hidden_layers, output_size, output_queue):
        # Import TensorFlow inside the subprocess
        import tensorflow as tf
        
        # Define the model architecture
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_size,), dtype=tf.float32))

        # Add hidden layers
        for layer_size in hidden_layers:
            model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

        # Output layer
        model.add(tf.keras.layers.Dense(output_size))

        model = tfmot.quantization.keras.quantize_model(model)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Save the model in TensorFlow's format
        model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
        model.save(model_path)

        output_queue.put(f"Model '{model_name}' created with QAT and saved at {model_path}")

    # Start a subprocess to create the model
    model_process = multiprocessing.Process(target=create_model_process, args=(model_name, input_size, hidden_layers, output_size, output_queue))
    model_process.start()
    model_process.join()  # Wait for the process to finish

    # Get the result from the queue
    result = output_queue.get()

    return jsonify({"message": result}), 200


@app.route('/update_tf_model', methods=['POST'])
def update_tf_model():
    data = request.get_json()

    model_name = data.get("model_name")
    new_x = np.array(data.get("features"), dtype=np.float32)
    new_y = np.array(data.get("labels"), dtype=np.float32)
    learning_rate = data.get("learning_rate", 0.001)  # Default to 0.001
    epochs = data.get("epochs", 5)  # Default to 5 epochs

    if not model_name or new_x is None or new_y is None:
        return jsonify({"error": "Missing required fields (model_name, features, labels)"}), 400

    # Create a multiprocessing queue to handle results from the subprocess
    output_queue = multiprocessing.Queue()
    
    # Import TensorFlow inside the subprocess
    import tensorflow as tf

    model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
    
    if not os.path.exists(model_path):
        output_queue.put(f"Model '{model_name}' not found")
        return

    # Load existing model
    model = tf.keras.models.load_model(model_path)

    print(model.summary())

    def update_model_process(model_name, new_x, new_y, learning_rate, epochs, output_queue):
        # Import TensorFlow inside the subprocess
        import tensorflow as tf

        model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
        
        if not os.path.exists(model_path):
            output_queue.put(f"Model '{model_name}' not found")
            return

        # Load existing model
        model = tf.keras.models.load_model(model_path)

        # Recompile with the user-specified learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                    loss='mean_squared_error')

        # Train with new data
        model.fit(new_x, new_y, epochs=epochs, batch_size=16, verbose=1)

        # Save the updated model
        model.save(model_path)

        output_queue.put(f"Model '{model_name}' updated with new training data!")

    # Start a subprocess to update the model
    update_process = multiprocessing.Process(target=update_model_process, args=(model_name, new_x, new_y, learning_rate, epochs, output_queue))
    update_process.start()
    update_process.join()  # Wait for the process to finish

    # Get the result from the queue
    result = output_queue.get()

    return jsonify({"message": result}), 200


@app.route('/compile_and_cocompile_models', methods=['POST'])
def compile_and_cocompile_models():
    data = request.json
    model_names = data.get("model_names", [])

    if not model_names:
        return jsonify({"error": "No model names provided"}), 400

    # Create a multiprocessing queue to handle results from the subprocess
    output_queue = multiprocessing.Queue()

    def compile_models_process(model_names, output_queue):
        # Import TensorFlow and other necessary modules inside the subprocess
        import tensorflow as tf

        tflite_model_paths = []

        for model_name in model_names:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.tf")
            
            if not os.path.exists(model_path):
                output_queue.put(f"Model '{model_name}' not found")
                return

            # Load the model
            model = tf.keras.models.load_model(model_path)

            # Define a representative dataset (for quantization)
            def representative_data_gen():
                input_shape = model.input_shape[1:]  # Get the input shape excluding the batch size
                for _ in range(300):  # Use 100 samples for the representative dataset
                    yield [np.random.uniform(0, 255, input_shape).astype(np.float32)]

            # Apply post-training quantization (int8 for Edge TPU compatibility)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.representative_dataset = representative_data_gen
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            tflite_model = converter.convert()

            # Save the TFLite model to a file
            tflite_model_path = os.path.join(MODEL_DIR, f"{model_name}.tflite")
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)

            tflite_model_paths.append(tflite_model_path)

        # Step 2: Send all TFLite models under "models" key for co-compilation
        files = [("models", (os.path.basename(tflite_model_path), open(tflite_model_path, 'rb'), "application/octet-stream"))
             for tflite_model_path in tflite_model_paths]

        # Send the request to the Coral Edge TPU compiler API
        response = requests.post(CORAL_EDGETPU_COMPILER_API_ADDRESS, files=files)

        if response.status_code == 200:
            # Save the compiled models as they are named
            zip_filename = "compiled_models.zip"
            with open(zip_filename, 'wb') as f:
                f.write(response.content)

            # Extract the zip file
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)

            output_queue.put("Models compiled and extracted successfully.")
        else:
            output_queue.put(f"Failed to co-compile models. Response: {response.status_code}")

    # Start a subprocess to compile the models
    compile_process = multiprocessing.Process(target=compile_models_process, args=(model_names, output_queue))
    compile_process.start()
    compile_process.join()  # Wait for the process to finish

    # Get the result from the queue
    result = output_queue.get()

    return jsonify({"message": result}), 200


@app.route('/run_inference', methods=['POST'])
def run_inference_on_coral():
    # Get the data from the request
    data = request.json
    model_name = data.get("model_name")
    inputs = np.array(data.get("features"), dtype=np.uint8)

    # Ensure that the inputs have the correct shape (e.g., [batch_size, input_size])
    input_shape = inputs.shape
    if len(input_shape) == 1:
        inputs = np.expand_dims(inputs, axis=0)  # Add batch dimension if needed

    # Create a multiprocessing queue to handle results from the subprocess
    output_queue = multiprocessing.Queue()

    def run_inference_process(model_name, inputs, output_queue):
        # Import pycoral and TensorFlow inside the subprocess
        import pycoral.utils.edgetpu as edgetpu

        # Check if Edge TPU library is available
        if os.path.exists('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1'):
            print("Edge TPU library found, using Edge TPU delegate.")
        else:
            print("Edge TPU library not found, fallback to CPU.")

        # Load the TFLite model with the Edge TPU delegate
        model_path = f"/app/models/{model_name}_edgetpu.tflite"  # Load the compiled model
        interpreter = edgetpu.make_interpreter(model_path)
        interpreter.allocate_tensors()

        # Get input and output tensor indices
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        results = []

        # Run inference for a batch of inputs
        for i in range(inputs.shape[0]):  # Process each input
            input_data = inputs[i]  # Get one input sample

            if len(input_data.shape) == 1:
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

            interpreter.set_tensor(input_index, input_data)

            # Run inference
            interpreter.invoke()

            # Get the output tensor
            output_data = interpreter.get_tensor(output_index)
            results.append(output_data.flatten().tolist())

        # Put the results into the output queue to return to Flask
        output_queue.put(results)

    # Start a subprocess to run inference
    inference_process = multiprocessing.Process(target=run_inference_process, args=(model_name, inputs, output_queue,))
    inference_process.start()
    inference_process.join()  # Wait for the process to finish

    # Get the output data from the queue
    output_data = output_queue.get()

    # Return the output data as a JSON response
    return jsonify({"output": output_data}), 200


if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=8000)
