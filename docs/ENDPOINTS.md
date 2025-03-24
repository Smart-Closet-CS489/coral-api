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

**Description:**
Creates a new tensorflow model with name `model_name` and size determined by `input_size`, `hidden_sizes`, and `output_size`. The activation functions used will be determined by the `model_type` parameter, which acts as a template. Finally, the `memory_size` parameter will determine how many training input/output pairs will be stored before the model starts to forget said data in new training sessions.

---

## Train a Model
**POST** `/models/{model_name}/training-session`

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

**Description:**
Starts a training session for model with name `model_name`. `Inputs` and `outputs` will be paired based on order (eg. the first input in `inputs` pairs with the first output in `outputs`). This new set of data is then mixed with memory input/output pairs, and this combined set of data is used to train the model. The memory data is refreshed several times throughout the training session so as to keep past learning relevant.

---

## See if a training session is active
**GET** `/models/{model_name}/training-session`

**Response Body:**
- `training_session_active` (boolean) – Specifies if there is an ongoing training session for this model

**Description:**
Model with name `model_name` can only have one training session active at a time; read the result from `training_session_active` to determine if a training session is currently active.

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

**Description:**
All models specified in `model_names` will be co-compiled together into their edgetpu tflite format. Doing this will allow the user to inference these models back-to-back without the models needed to be switched in SRAM, improving performance. Do this for groups of models that will be inferenced together, or a single model if independent. Remember, the max SRAM size is 7MB, so co-compiling models with a combined size larger than 7MB will force inefficient CPU usage.

**Important Things to Note:**
- Total number of datapoints for any training session (input/output pairs) must be at least 6
- Input/output indidivual size must match the number of input and output neurons for the specific model

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

**Description:**
Loads the model with name `model_name` into the Google Coral (if not already present), and then passes all `inputs` into the coral to be inferenced. The `outputs` (in the same order of the inputs) are then returned in the request body.

---

## Delete a Model
**DELETE** `/models/{model_name}`

**Description:**
Deletes the model with name `model_name` and all associated information (if it exists).

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

**Description:**
Returns all information for the model with name `model_name`.
