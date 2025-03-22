# api_helper.py

# **** I need this file becuase of conflicting imports ********************************

# ---- Imports -------------------------------
from flask import Flask
from .resources.inference import post_inference

# ---- App creation ------------------------------
app_helper = Flask(__name__)

# ---- Registering the endpoints ------------------
app_helper.add_url_rule('/models/<model_name>/inference', 'post_inference', post_inference, methods=['POST'])
