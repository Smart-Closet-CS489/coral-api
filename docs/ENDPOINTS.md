# API Endpoints Documentation

## Create a Model
**POST** `/models`

**Request Body:**
- `model_name` (string) – Name of the model
- `input_size` (int) – Number of input features
- `hidden_sizes` (list of ints) – List of hidden layer sizes
- `output_size` (int) – Number of output features
- `model_type` (string) – Type of model ("regression", "binary", "classification")
- `memory_size` (int) – Memory size for history buffer

---

## Train a Model
**POST** `/models/{model_name}/training_session`

**Request Body:**
- `inputs` (list of lists) – List of input values
- `outputs` (list of lists) – Corresponding output values

---

## Compile Model for Edge TPU
**POST** `/models/compilations`

**Request Body:**
- `model_names` (list of strings) – Names of models to compile

---

## Run Inference
**POST** `/models/{model_name}/inference`

**Request Body:**
- `inputs` (list of lists) – List of input values for inference

**Response Body:**
- `outputs` (list of lists) – List of output values returned from the inference

---

## Delete a Model
**DELETE** `/models/{model_name}`

---

## Get Model Information
**GET** `/models/{model_name}`

**Response Body:**
- `model_name` (string) – Name of the model
- `input_size` (int) – Number of input features
- `hidden_sizes` (list of ints) – List of hidden layer sizes
- `output_size` (int) – Number of output features
- `model_type` (string) – Type of model ("regression", "binary", "classification")
- `memory_size` (int) – Memory size for history buffer
