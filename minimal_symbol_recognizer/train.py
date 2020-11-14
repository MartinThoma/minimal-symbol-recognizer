# Core Library modules
import csv
from pathlib import Path
from typing import Dict


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
    print(path2label)


def read_labels(labels_path: Path) -> Dict[Path, str]:
    path2label = {}
    with open(labels_path) as fp:
        reader = csv.reader(fp, delimiter=",")
        for row in reader:
            path = (labels_path.parent / row[0]).absolute
            path2label[path] = row[2]
    return path2label
