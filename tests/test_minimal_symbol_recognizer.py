#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `minimal_symbol_recognizer` package."""

# Third party modules
import pytest

# First party modules
import minimal_symbol_recognizer


def test_version():
    assert minimal_symbol_recognizer.__version__.count(".") == 2


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
