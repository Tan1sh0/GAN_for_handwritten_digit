# GAN_for_handwritten_digit

Using a GAN model to generate real looking handwritten numbers.

## Generate a CAPTCHA-style image

Install TensorFlow and Pillow in the Python environment used to run the script:

```powershell
python -m pip install tensorflow pillow
```

Run the example below from the project folder. The saved generator file,
`cgan_mnist_generator.keras`, should be beside `captcha_generator.py` by default:

```powershell
python captcha_generator.py 4827 --output sample_captcha.png --seed 123
```

The script saves a four-digit grayscale PNG. The seed is optional; use the same
seed, code, model, rotation, and line settings to reproduce an image. Rotation
and interference lines can be adjusted with `--rotation` (0-15 degrees) and
`--lines` (non-negative count). Pass `--model PATH` if the model is elsewhere.

The `generate_captcha` function can also be imported and called from Python:

```python
from captcha_generator import generate_captcha

image = generate_captcha(
    "4827",
    output_path="sample_captcha.png",
    seed=123,
)
```

## Run the CAPTCHA user-testing web app

Install the project dependencies into your environment and start the local
Flask app:

```powershell
python -m pip install -r requirements.txt
python webapp.py
```

Open <http://127.0.0.1:5000> in a browser. The page generates a four-digit
challenge, accepts an answer, and reports whether it matches. Use “Get a
different challenge” or “Try another challenge” to continue. Challenges are
kept in process memory for up to 15 minutes and are not recorded; the app is a
single-process local testing tool.

This generated MNIST CAPTCHA is for usability testing only. It is not a
production anti-bot security control.
