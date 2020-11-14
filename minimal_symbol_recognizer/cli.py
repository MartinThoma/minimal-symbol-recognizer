#!/usr/bin/env python

# Third party modules
import click

# First party modules
import minimal_symbol_recognizer


@click.group()
@click.version_option(version=minimal_symbol_recognizer.__version__)
def entry_point():
    """Awesomeproject spreads pure awesomeness."""
