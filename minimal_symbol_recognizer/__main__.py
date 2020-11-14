"""
This file exists to make the live of people without a propper set PATH easier.

It allows them to execute the module like this:

    $ python -m minimal_symbol_recognizer --help
"""

# First party modules
import minimal_symbol_recognizer.cli

minimal_symbol_recognizer.cli.entry_point()
