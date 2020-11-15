# Core Library modules
from pathlib import Path
from typing import Dict, List, Tuple

# Third party modules
import numpy as np
from keras import Sequential
from keras.models import load_model as load_keras_model
from PIL import Image

# First party modules
from minimal_symbol_recognizer.preprocess import preprocess

model = None
labels = None


def predict(
    model_path: Path, labels_path: Path, image: Image
) -> List[Tuple[str, float]]:
    model, labels = load_model(model_path, labels_path)
    nb_images = 1
    width = 32
    height = 32
    images = np.zeros((nb_images, width, height))
    images[0, :, :] = preprocess(image)
    images = np.expand_dims(images, -1)
    prediction = model.predict(images)
    cls2prob = get_class_probabilities(prediction[0], labels)
    return order_by_prob(cls2prob)


def load_model(model_path: Path, labels_path: Path) -> Sequential:
    if globals()["model"] is None:
        loaded_model = load_keras_model(model_path)
        globals()["model"] = loaded_model
        globals()["labels"] = load_labels(labels_path)
    return globals()["model"], globals()["labels"]


def load_labels(labels_path: Path) -> List[str]:
    with open(labels_path) as fp:
        data = fp.read().strip()
    return data.split("\n")


def get_class_probabilities(pred: np.array, class_list: List[str]) -> Dict[str, float]:
    return {class_: float(prob) for prob, class_ in zip(pred, class_list)}


def order_by_prob(cls2prob: Dict[str, float]) -> List[Tuple[str, float]]:
    items = list(cls2prob.items())
    return sorted(items, key=lambda n: n[1], reverse=True)
