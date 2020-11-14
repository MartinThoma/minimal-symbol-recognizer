#!/usr/bin/env python

# Core Library modules
from pathlib import Path

# Third party modules
import click

# First party modules
import minimal_symbol_recognizer
from minimal_symbol_recognizer.app import run_test_server
from minimal_symbol_recognizer.train import main as train_main


@click.group()
@click.version_option(version=minimal_symbol_recognizer.__version__)
def entry_point():
    """Symbol recognition project to learn Python"""


@entry_point.command()
@click.option(
    "--in",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--out",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
)
def train(input_path: str, output_path: str) -> None:
    train_main(Path(input_path), Path(output_path))


@entry_point.command()
@click.option(
    "--model",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
)
def run_server(model: str) -> None:
    run_test_server(Path(model))
