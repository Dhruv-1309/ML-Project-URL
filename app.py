"""Flask frontend for the phishing URL detector."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

from model_service import MODEL_PATH, load_model_metrics, predict_url, predict_urls


app = Flask(__name__)
PUBLIC_DIR = Path(__file__).resolve().parent / "public"


@app.route("/style.css")
def public_style():
    """Serve local stylesheet during Flask development."""
    return send_from_directory(PUBLIC_DIR, "style.css")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    batch_results = []
    error = None
    input_url = ""
    batch_input = ""
    model_ready = Path(MODEL_PATH).exists()
    model_metrics = load_model_metrics() if model_ready else {}

    if request.method == "POST":
        mode = request.form.get("mode", "single")
        input_url = request.form.get("url", "").strip()
        batch_input = request.form.get("batch_urls", "").strip()

        if not model_ready:
            error = (
                "Model file not found. Train the model first by running "
                "`python train_model.py` after placing `dataset.csv` in the project root."
            )
        else:
            try:
                if mode == "batch":
                    batch_results = predict_urls(batch_input.splitlines())
                else:
                    result = predict_url(input_url)
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        input_url=input_url,
        batch_input=batch_input,
        batch_results=batch_results,
        model_ready=model_ready,
        model_metrics=model_metrics,
    )


if __name__ == "__main__":
    app.run(debug=True)
