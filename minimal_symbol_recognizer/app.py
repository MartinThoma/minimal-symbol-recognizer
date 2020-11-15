"""Run a Flask web server for symbol recognition."""

# Core Library modules
from pathlib import Path

# Third party modules
from flask import Flask, render_template

model = None


def load_model(model_path: Path) -> None:
    loaded_model = model_path  # TODO
    globals()["model"] = loaded_model


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


def run_test_server(model: Path) -> None:
    app = create_app()
    app.run(host="0.0.0.0")
