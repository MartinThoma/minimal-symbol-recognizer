# minimal_symbol_recognizer

Train and use a classifier for handwritten symbols.

## Installation

```
$ pip install git+https://github.com/MartinThoma/minimal-symbol-recognizer
```

## Usage

Download the [`HASYv2.tar.bz2`](https://zenodo.org/record/259444). Then:

```
# Generate a neural network model and the labels.csv
$ minimal_symbol_recognizer train --in HASYv2.tar.bz2 --out model.h5

# Use it to classify symbols. This will start a local web server.
# You can go to http://0.0.0.0:5000/ and should see something similar to
# write-math.com
$ minimal_symbol_recognizer run-server --model model.h5 --labels labels.csv
```

You can always use the help function:

```
minimal_symbol_recognizer --help
Usage: minimal_symbol_recognizer [OPTIONS] COMMAND [ARGS]...

  Symbol recognition project to learn Python

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  run-server  Start a local Flask development server to use the model.
  train       Train a symbol recognition model.
```

### Usage with Docker

[Install Docker](https://docs.docker.com/get-docker/) first. Then go into this directory and
run:

```
$ docker build -t symbol_recognizer_service .
$ docker run -p 5000:5000 symbol_recognizer_service
```

You might need `sudo` (root privileges) to run Docker. [More information about Docker](https://martin-thoma.com/docker/)
