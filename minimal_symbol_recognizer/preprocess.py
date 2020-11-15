"""Preprocessing steps for image data."""

# Core Library modules
import math
from typing import NamedTuple

# Third party modules
import numpy as np
from PIL import Image


class BoundingBox(NamedTuple):
    min_x: int
    max_x: int
    min_y: int
    max_y: int


def preprocess(image: Image) -> np.array:
    image = to_grayscale(image)
    image = crop_to_content(image)
    image = fix_aspect_ratio(image)
    image = scale(image, width=32, height=32)
    image.save("preprocessed.png")
    image_array = np.array(image, dtype=np.float32) / 255
    return image_array


def to_grayscale(image: Image) -> Image:
    if image.mode == "RGBA":
        white_in_rgb = (255, 255, 255)
        background = Image.new("RGB", image.size, white_in_rgb)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = background
    return image.convert("L")


def crop_to_content(image: Image) -> Image:
    bb = get_bounding_box(image)
    cropped_example = image.crop((bb.min_x, bb.min_y, bb.max_x, bb.max_y))
    return cropped_example


def fix_aspect_ratio(image: Image) -> Image:
    expected = max(image.width, image.height)
    white_in_l = 255
    new_im = Image.new("L", (expected, expected), white_in_l)
    x_add = int(math.floor((expected - image.width) / 2))
    y_add = int(math.floor((expected - image.height) / 2))
    new_im.paste(image, (x_add, y_add))
    return new_im


def scale(image: Image, width: int, height: int) -> Image:
    resized = image.resize((width, height))
    return resized


def get_bounding_box(image: Image) -> BoundingBox:
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for x in range(0, image.width):
        for y in range(image.height):
            if (
                image.getpixel((x, y)) < 254
            ):  # note that the first parameter is actually a tuple object
                if min_x is None:
                    min_x = x
                    max_x = x
                    min_y = y
                    max_y = y
                else:
                    min_x = min(x, min_x)
                    max_x = max(x, max_x)
                    min_y = min(y, min_y)
                    max_y = max(y, max_y)
    if min_x is None or max_x is None or min_y is None or max_y is None:
        raise ValueError("Image was empty")
    return BoundingBox(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
