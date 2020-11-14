# minimal_symbol_recognizer

Train and use a classifier for handwritten symbols.

## Installation

```
$ pip install git+https://github.com/MartinThoma/minimal-symbol-recognizer
```

## Usage

Download the [`HASYv2.tar.bz2`](https://zenodo.org/record/259444). Then:

```
$ minimal_symbol_recognizer train --in HASYv2.tar.bz2 --out model.h5
$ minimal_symbol_recognizer run-server --model model.h5
```
