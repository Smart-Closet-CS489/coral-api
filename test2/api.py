from flask import Flask, jsonify, request
import numpy as np
import os
import multiprocessing
import requests

app = Flask(__name__)

compiler_url = "http://134.199.176.64/compile"

# Step 1: Create a small model with random data using TensorFlow
def create_model_process():
    import tensorflow as tf
    
    # Define and create the model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10,), dtype=tf.float32),  # Specify input type as float32 (you can also use int8 if you want to directly use quantized inputs)
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define a representative dataset (for quantization)
    def representative_data_gen():
        # Generate a small sample of input data that represents the distribution
        for _ in range(100):  # Use 100 samples for the representative dataset
            yield [np.random.random((10,)).astype(np.float32)]

    # Apply post-training quantization (int8 for Edge TPU compatibility)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_data_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    model_path = '/app/model.tflite'
    with open(model_path, 'wb') as f:
        f.write(tflite_model)

    # STARTING HERE
    with open(model_path, 'rb') as model_file:
        # Prepare the files to send in the POST request
        files = {'model': model_file}
        
        # Send the POST request to compile the model
        response = requests.post(compiler_url, files=files)

        # Check the response from the server
        if response.status_code == 200:
            # Use a consistent file name for the compiled model
            compiled_model_path = '/app/compiled_model.tflite'
            
            # Save the compiled model with a consistent file name
            with open(compiled_model_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Model compiled and saved as '{compiled_model_path}'.")
        else:
            print(f"Error: {response.status_code}, {response.json()}")
    # ENDING HERE

    return "Model created and converted to TFLite!"

# Step 2: Run the TFLite model on the Coral USB Accelerator using TensorFlow Lite API
def run_inference_process(output_queue):
    # Import pycoral utils only when needed
    import pycoral.utils.edgetpu as edgetpu

    if os.path.exists('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1'):
        print("Edge TPU library found, using Edge TPU delegate.")
    else:
        print("Edge TPU library not found, fallback to CPU.")

    # Load the TFLite model with the Edge TPU delegate
    model_path = '/app/compiled_model.tflite'  # Load the compiled model
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Generate some random input data
    input_data = np.random.random((1, 10)).astype(np.uint8)

    # Get input and output tensor indices
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    results = []

    # Run the inference 100 times
    for _ in range(4000):
        # Set the input tensor
        interpreter.set_tensor(input_index, input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_index)
        results.append(output_data.tolist())

    # Put the results into the output queue to return to Flask
    output_queue.put(results)

# Step 1: Create a small model with random data using TensorFlow
@app.route('/create_model', methods=['POST'])
def create_model():
    # Use multiprocessing to run model creation in a separate process
    model_process = multiprocessing.Process(target=create_model_process)
    model_process.start()
    model_process.join()  # Wait for the model creation process to finish

    return jsonify({"message": "Model created and converted to TFLite!"}), 200

# Step 2: Run the TFLite model on the Coral USB Accelerator using TensorFlow Lite API
@app.route('/run_inference', methods=['POST'])
def run_inference_on_coral():
    # Create a queue to capture the output of the inference process
    output_queue = multiprocessing.Queue()

    # Use multiprocessing to run inference in a separate process
    inference_process = multiprocessing.Process(target=run_inference_process, args=(output_queue,))
    inference_process.start()
    inference_process.join()  # Wait for the inference process to finish

    # Get the output data from the queue
    output_data = output_queue.get()

    return jsonify({"output": output_data}), 200


if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000)
