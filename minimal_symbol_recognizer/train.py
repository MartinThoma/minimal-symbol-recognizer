# Core Library modules
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
    paths = list(path2label.keys())
    labels = [path2label[path] for path in paths]
    label2index = get_indices(set(labels))
    nb_classes = len(label2index)
    images = read_images(paths)
    width = images.shape[1]
    height = images.shape[2]
    model = create_model((width, height, 1), nb_classes)
    print(model.summary())
    x_train = images / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = np.array([label2index[label] for label in labels])
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    train_model(model, x_train, y_train)
    model.save(output_file)


def read_labels(labels_path: Path) -> Dict[Path, str]:
    path2label = {}
    with open(labels_path) as fp:
        reader = csv.reader(fp, delimiter=",")
        next(reader)  # skip header
        for row in reader:
            path = (labels_path.parent / row[0]).resolve()
            path2label[path] = row[2]
    return path2label


def read_images(image_paths: List[Path]) -> np.ndarray:
    nb_images = len(image_paths)
    width = 32
    height = 32
    images = np.zeros((nb_images, width, height))
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path).convert(mode="L") as im:
            im_arr = np.array(im, dtype=np.float)
            images[index, :, :] = im_arr
        if index % 1000 == 0:
            print(f"{index/nb_images*100:3.0f}%: {index:,} out of {nb_images:,} done")
    return images


def get_indices(labels: Set[str]) -> Dict[str, int]:
    labels_list = sorted(labels)
    return {label: index for index, label in enumerate(labels_list)}


def create_model(
    input_shape: Tuple[int, int, int], num_classes: int
) -> keras.Sequential:
    # See https://keras.io/examples/vision/mnist_convnet/
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def train_model(
    model: keras.Sequential, x_train: np.ndarray, y_train: np.ndarray
) -> None:
    batch_size = 128
    epochs = 15

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        shuffle=True,
    )
