"""Run a Flask web server for symbol recognition."""

# Core Library modules
import base64
import io
from pathlib import Path
from typing import Any, Dict

# Third party modules
from flask import Flask, render_template, request
from PIL import Image

# First party modules
from minimal_symbol_recognizer.predict import predict


def create_app(model: Path, labels: Path) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/classify", methods=["POST"])
    def classify() -> Dict[str, Any]:
        imagestr = request.form["imgBase64"]
        decoded = base64.b64decode(imagestr.split(",")[1])
        image = Image.open(io.BytesIO(decoded))
        predictions = predict(model, labels, image)
        image.close()
        for pred in predictions[:5]:
            print(pred)  # TODO: Just temporarily added
        return {
            "errors": [],
            "prediction": [
                {"symbol": pred, "probability": f"{prob * 100:.0f}%"}
                for pred, prob in predictions[:5]
            ],
        }

    return app


def run_test_server(model: Path, labels: Path) -> None:
    app = create_app(model, labels)
    app.run(host="0.0.0.0")
