from pathlib import Path

import joblib
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.sav"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

app = Flask(__name__)


def load_artifact(path):
    if not path.exists():
        return None
    return joblib.load(path)


model = load_artifact(MODEL_PATH)
vectorizer = load_artifact(VECTORIZER_PATH)


@app.route("/")
def home():
    setup_error = None

    if model is None:
        setup_error = "trained_model.sav was not found."
    elif vectorizer is None:
        setup_error = (
            "vectorizer.pkl is missing. Save the fitted TfidfVectorizer from the "
            "training notebook and place it in this folder."
        )

    return render_template("index.html", setup_error=setup_error)


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    if not message:
        return render_template("index.html", error="Please enter a message.")

    if model is None:
        return render_template("index.html", setup_error="trained_model.sav was not found.")

    if vectorizer is None:
        return render_template(
            "index.html",
            message=message,
            setup_error=(
                "vectorizer.pkl is missing. The model expects vectorized text, so "
                "predictions cannot run until the fitted vectorizer is added."
            ),
        )

    transformed_message = vectorizer.transform([message])
    prediction = model.predict(transformed_message)[0]

    result = "Spam Message" if prediction == 0 else "Ham Message"

    return render_template("index.html", message=message, prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
