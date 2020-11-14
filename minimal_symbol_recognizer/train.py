# Core Library modules
from pathlib import Path


def main(input_file: Path, output_file: Path) -> None:
    """
    Train a model to recognize a symbol.

    Parameters
    ----------
    input_file: Path
        Path to the downloaded https://zenodo.org/record/259444 dataset.
    output_file: Path
        Path where the trained model is stored.
    """
