# inference.py

# ---- Imports ----------------------------------
from flask import Flask
from .endpoints import run

# ---- App creation ------------------------------
app = Flask(__name__)

# ---- Registering the endpoints ------------------
app.add_url_rule('/inference/run', 'run', run, methods=['POST'])