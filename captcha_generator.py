"""Generate four-digit CAPTCHA-style images from the saved MNIST cGAN."""

import argparse
import random
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name(
    "cgan_mnist_generator.keras"
)
CELL_SIZE = (52, 60)
CELL_INSET = 6
CELL_GAP = 4
IMAGE_PADDING = 8


def _make_digit_cell(
    generated: np.ndarray,
    rng: random.Random,
    max_rotation: float,
) -> Image.Image:
    """Convert one tanh-scaled model output to a proportionally fitted cell."""
    pixels = np.asarray(generated, dtype=np.float32)
    if pixels.shape != (28, 28):
        raise ValueError(
            f"Expected a generated digit with shape (28, 28), got {pixels.shape}."
        )

    grayscale = np.rint(np.clip((pixels + 1.0) * 127.5, 0, 255)).astype(
        np.uint8
    )
    digit = Image.fromarray(255 - grayscale)
    foreground = digit.point(lambda value: 255 if value > 16 else 0)
    bounds = foreground.getbbox()
    if bounds is None:
        raise RuntimeError("The generator produced a blank digit image.")
    digit = digit.crop(bounds)

    cell_width, cell_height = CELL_SIZE
    available_width = cell_width - 2 * CELL_INSET
    available_height = cell_height - 2 * CELL_INSET
    scale = min(
        available_width / digit.width,
        available_height / digit.height,
    )
    resized_size = (
        max(1, round(digit.width * scale)),
        max(1, round(digit.height * scale)),
    )
    digit = digit.resize(resized_size, Image.Resampling.LANCZOS)

    cell = Image.new("L", CELL_SIZE, color=255)
    position = (
        (cell_width - digit.width) // 2,
        (cell_height - digit.height) // 2,
    )
    cell.paste(digit, position)

    angle = rng.uniform(-max_rotation, max_rotation)
    return cell.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )


def generate_captcha(
    code: str,
    output_path: Optional[Union[str, Path]] = "captcha.png",
    model_path: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    max_rotation: float = 6.0,
    interference_lines: int = 2,
    generator: Optional[tf.keras.Model] = None,
) -> Image.Image:
    """Generate and save a four-digit CAPTCHA-style image.

    The returned image is grayscale. Each generated digit is scaled without
    changing its aspect ratio and centered in its own cell.
    """
    if not isinstance(code, str) or re.fullmatch(r"[0-9]{4}", code) is None:
        raise ValueError("code must be a string containing exactly four digits.")
    if not 0 <= max_rotation <= 15:
        raise ValueError("max_rotation must be between 0 and 15 degrees.")
    if not isinstance(interference_lines, int) or interference_lines < 0:
        raise ValueError("interference_lines must be a non-negative integer.")

    if generator is None:
        resolved_model_path = (
            Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        )
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Generator model not found: {resolved_model_path}"
            )
        generator = tf.keras.models.load_model(
            resolved_model_path,
            compile=False,
        )
    labels = tf.keras.utils.to_categorical(
        [int(digit) for digit in code],
        num_classes=10,
    ).astype(np.float32)
    if seed is None:
        noise = tf.random.normal((4, 100), dtype=tf.float32)
    else:
        noise = tf.random.Generator.from_seed(seed).normal(
            (4, 100),
            dtype=tf.float32,
        )

    generated = generator(
        [noise, tf.convert_to_tensor(labels)],
        training=False,
    )
    generated = np.asarray(generated)
    if generated.shape != (4, 28, 28, 1):
        raise ValueError(
            "The model must return images with shape (4, 28, 28, 1); "
            f"got {generated.shape}."
        )

    rng = random.Random(seed)
    cells = [
        _make_digit_cell(generated[index, :, :, 0], rng, max_rotation)
        for index in range(4)
    ]
    cell_width, cell_height = CELL_SIZE
    image_width = (
        2 * IMAGE_PADDING + 4 * cell_width + 3 * CELL_GAP
    )
    image_height = 2 * IMAGE_PADDING + cell_height
    image = Image.new("L", (image_width, image_height), color=255)
    for index, cell in enumerate(cells):
        x = IMAGE_PADDING + index * (cell_width + CELL_GAP)
        image.paste(cell, (x, IMAGE_PADDING))

    if interference_lines:
        overlay = Image.new("L", image.size, color=255)
        draw = ImageDraw.Draw(overlay)
        line_rng = np.random.default_rng(seed)
        for _ in range(interference_lines):
            start = (
                int(line_rng.integers(0, image_width)),
                int(line_rng.integers(0, image_height)),
            )
            end = (
                int(line_rng.integers(0, image_width)),
                int(line_rng.integers(0, image_height)),
            )
            draw.line((start, end), fill=150, width=1)
        image = Image.blend(image, overlay, 0.2)

    if output_path is not None:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(resolved_output_path, format="PNG")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a four-digit CAPTCHA-style image using the saved cGAN."
        )
    )
    parser.add_argument("code", help="Exactly four digits, for example 4827.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("captcha.png"),
        help="PNG output path (default: captcha.png).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Generator model path (default: cgan_mnist_generator.keras "
            "beside this script)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible digits and effects.",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=6.0,
        help="Maximum absolute rotation in degrees (0-15, default: 6).",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=2,
        help="Number of mild interference lines (default: 2).",
    )
    args = parser.parse_args()
    generate_captcha(
        args.code,
        output_path=args.output,
        model_path=args.model,
        seed=args.seed,
        max_rotation=args.rotation,
        interference_lines=args.lines,
    )
    print(f"Saved CAPTCHA image to {args.output}")


if __name__ == "__main__":
    main()
