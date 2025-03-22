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

**Sample Request Body:**
```
{
    "model_name": "sample_model",
    "input_size": 2,
    "hidden_sizes": [128, 128, 128, 128],
    "output_size": 1,
    "model_type": "regression",
    "memory_size": 1000
}
```

---

## Train a Model
**POST** `/models/{model_name}/training_session`

**Request Body:**
- `inputs` (list of lists) – List of input values
- `outputs` (list of lists) – Corresponding output values

**Sample Request Body:**
```
{
"inputs": [[0.2, 0.2], [0.75, 0.75], [0.33, 0.33]],
"outputs": [[0.2], [0.75], [0.33]]
}
```

---

## Compile Model for Edge TPU
**POST** `/models/compilations`

**Request Body:**
- `model_names` (list of strings) – Names of models to compile (or co-compile if multiple)

**Sample Request Body:**
```
{
    "model_names": ["sample_model", "test_model"]
}
```

---

## Run Inference
**POST** `/models/{model_name}/inference`

**Request Body:**
- `inputs` (list of lists) – List of input values for inference

**Sample Request Body:**
```
{
    "inputs": [[0.2, 0.2], [0.75, 0.75], [0.33, 0.33], [0.45, 0.45]]
}
```

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
