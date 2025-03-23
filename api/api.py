#api.py

# ---- Imports -----------------------------------
from flask import Flask
from .resources.models import post_model, get_model, delete_model
from .resources.compilations import post_compilation
from .resources.training_session import post_training_session, get_training_session

# ---- App creation ------------------------------
app = Flask(__name__)

# ---- Registering the endpoints ------------------
app.add_url_rule('/models', 'post_model', post_model, methods=['POST'])
app.add_url_rule('/models/<model_name>', 'get_model', get_model, methods=['GET'])
app.add_url_rule('/models/<model_name>', 'delete_model', delete_model, methods=['DELETE'])
app.add_url_rule('/models/compilations', 'post_compilation', post_compilation, methods=['POST'])
app.add_url_rule('/models/<model_name>/training_session', 'post_training_session', post_training_session, methods=['POST'])
app.add_url_rule('/models/<model_name>/training_session', 'get_training_session', get_training_session, methods=['GET'])
