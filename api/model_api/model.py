# model.py

# ---- Imports ----------------------------------
from flask import Flask
from .endpoints import create, compile_edgetpu, train

# ---- App creation ------------------------------
app = Flask(__name__)

# ---- Registering the endpoints ------------------
app.add_url_rule('/model/create', 'create', create, methods=['POST'])
app.add_url_rule('/model/train', 'train', train, methods=['POST'])
app.add_url_rule('/model/compile_edgetpu', 'compile_edgetpu', compile_edgetpu, methods=['POST'])