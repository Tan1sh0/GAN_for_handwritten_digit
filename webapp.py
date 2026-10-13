"""Local Flask app for testing the generated CAPTCHA with human users."""

import io
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import tensorflow as tf
from flask import Flask, abort, make_response, render_template, request, send_file, session

from captcha_generator import DEFAULT_MODEL_PATH, generate_captcha


CHALLENGE_TTL_SECONDS = 15 * 60


@dataclass(frozen=True)
class Challenge:
    answer: str
    image: bytes
    created_at: float


def create_app(
    model_path: Path = DEFAULT_MODEL_PATH,
    model_loader: Callable[..., tf.keras.Model] = tf.keras.models.load_model,
) -> Flask:
    """Create the app; challenge records are held in process memory."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32)),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    challenges: dict[str, Challenge] = {}
    challenge_lock = threading.Lock()
    model: Optional[tf.keras.Model] = None
    model_lock = threading.Lock()
    app.extensions["captcha_challenges"] = challenges

    def get_model() -> tf.keras.Model:
        nonlocal model
        if model is None:
            with model_lock:
                if model is None:
                    if not model_path.is_file():
                        raise FileNotFoundError(
                            f"Generator model not found: {model_path}"
                        )
                    model = model_loader(model_path, compile=False)
        return model

    def issue_challenge() -> str:
        answer = f"{secrets.randbelow(10_000):04d}"
        challenge_image = generate_captcha(
            answer,
            output_path=None,
            generator=get_model(),
        )
        image_buffer = io.BytesIO()
        challenge_image.save(image_buffer, format="PNG")
        token = secrets.token_urlsafe(32)
        now = time.monotonic()
        with challenge_lock:
            expired = [
                key
                for key, value in challenges.items()
                if now - value.created_at > CHALLENGE_TTL_SECONDS
            ]
            for key in expired:
                del challenges[key]
            challenges[token] = Challenge(
                answer=answer,
                image=image_buffer.getvalue(),
                created_at=now,
            )
        session["challenge_token"] = token
        return token

    @app.get("/")
    def index():
        token = issue_challenge()
        return render_template("index.html", image_url=f"/captcha/{token}.png")

    @app.get("/captcha/<token>.png")
    def captcha_image(token: str):
        if not secrets.compare_digest(
            token,
            session.get("challenge_token", ""),
        ):
            abort(404)
        with challenge_lock:
            challenge = challenges.get(token)
        if challenge is None:
            abort(404)
        if time.monotonic() - challenge.created_at > CHALLENGE_TTL_SECONDS:
            with challenge_lock:
                challenges.pop(token, None)
            abort(404)

        response = make_response(
            send_file(
                io.BytesIO(challenge.image),
                mimetype="image/png",
                download_name="captcha.png",
            )
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/answer")
    def check_answer():
        token = session.get("challenge_token", "")
        with challenge_lock:
            challenge = challenges.get(token)
        if challenge is None:
            session.pop("challenge_token", None)
            return render_template(
                "result.html",
                message="That challenge has expired. Start a new one.",
                correct=False,
            ), 410
        if time.monotonic() - challenge.created_at > CHALLENGE_TTL_SECONDS:
            with challenge_lock:
                challenges.pop(token, None)
            session.pop("challenge_token", None)
            return render_template(
                "result.html",
                message="That challenge has expired. Start a new one.",
                correct=False,
            ), 410

        answer = request.form.get("answer", "")
        if re.fullmatch(r"[0-9]{4}", answer) is None:
            return render_template(
                "index.html",
                image_url=f"/captcha/{token}.png",
                error="Enter exactly four digits.",
            ), 400

        with challenge_lock:
            challenges.pop(token, None)
        session.pop("challenge_token", None)
        correct = secrets.compare_digest(answer, challenge.answer)
        message = "Correct — thanks for testing!" if correct else "Not quite. Try another challenge."
        return render_template(
            "result.html",
            message=message,
            correct=correct,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
