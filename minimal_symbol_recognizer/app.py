"""Run a Flask web server for symbol recognition."""

# Core Library modules
import base64
from pathlib import Path
from typing import Any, Dict

# Third party modules
from flask import Flask, render_template, request

model = None


def load_model(model_path: Path) -> None:
    loaded_model = model_path  # TODO
    globals()["model"] = loaded_model


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/classify", methods=["POST"])
    def classify() -> Dict[str, Any]:
        imagestr = request.form["imgBase64"]
        with open("example.png", "wb") as fp:
            decoded = base64.b64decode(imagestr.split(",")[1])
            fp.write(decoded)
        return {"errors": []}

    return app


def run_test_server(model: Path) -> None:
    app = create_app()
    app.run(host="0.0.0.0")
