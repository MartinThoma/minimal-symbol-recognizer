# Core Library modules
import csv
from pathlib import Path
from typing import Dict, Tuple

# Third party modules
import keras
import numpy as np
from PIL import Image
from tensorflow.keras import layers


def main(input_directory: Path, output_file: Path) -> None:
    """
    Train a model to recognize a symbol.

    Parameters
    ----------
    input_directory: Path
        Path to the downloaded https://zenodo.org/record/259444 dataset.
    output_file: Path
        Path where the trained model is stored.
    """
    path2label = read_labels(input_directory / "hasy-data-labels.csv")
    nb_classes = len(set(path2label.values()))
    images = read_images(input_directory / "hasy-data")
    width = images.shape[1]
    height = images.shape[2]
    model = create_model((width, height, 1), nb_classes)
    print(model.summary())


def read_labels(labels_path: Path) -> Dict[Path, str]:
    path2label = {}
    with open(labels_path) as fp:
        reader = csv.reader(fp, delimiter=",")
        for row in reader:
            path = (labels_path.parent / row[0]).absolute
            path2label[path] = row[2]
    return path2label


def read_images(images_directory: Path) -> np.ndarray:
    filepaths = list(images_directory.glob("*.png"))
    nb_images = len(filepaths)
    width = 32
    height = 32
    images = np.zeros((nb_images, width, height))
    for index, image_path in enumerate(filepaths):
        with Image.open(image_path).convert(mode="L") as im:
            im_arr = np.array(im)
            images[index, :, :] = im_arr
        if index % 1000 == 0:
            print(f"{index/nb_images*100:3.0f}%: {index:,} out of {nb_images:,} done")
    return images


def create_model(
    input_shape: Tuple[int, int, int], num_classes: int
) -> keras.Sequential:
    # See https://keras.io/examples/vision/mnist_convnet/
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
