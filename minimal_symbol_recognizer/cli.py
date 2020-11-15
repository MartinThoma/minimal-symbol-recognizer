#!/usr/bin/env python

# Core Library modules
from pathlib import Path

# Third party modules
import click

# First party modules
import minimal_symbol_recognizer
from minimal_symbol_recognizer.app import run_test_server


@click.group()
@click.version_option(version=minimal_symbol_recognizer.__version__)
def entry_point():
    """Symbol recognition project to learn Python"""


@entry_point.command()
@click.option(
    "--in",
    "input_path",
    # metavar="HASYv2-Directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
    help="The HASYv2 directory, extracted from the HASYv2.tar.gz",
)
@click.option(
    "--out",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    required=True,
    help="The trained model",
)
def train(input_path: str, output_path: str) -> None:
    """Train a symbol recognition model."""
    # First party modules
    from minimal_symbol_recognizer.train import main as train_main

    train_main(Path(input_path), Path(output_path))


@entry_point.command()
@click.option(
    "--model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help="A model file. This could be the output of the 'train' subcommand",
)
def run_server(model: str) -> None:
    """Start a local Flask development server to use the model."""
    run_test_server(Path(model))
