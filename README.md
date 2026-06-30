# Spam Detection Project

Flask web app for a saved scikit-learn spam classifier.

## Project Structure

```text
SpamDetectionProject/
├── app.py
├── trained_model.sav
├── vectorizer.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md
```

## Important

`trained_model.sav` is included, but the app also needs the fitted preprocessing object used during training.

If your notebook used `TfidfVectorizer`, save it after fitting:

```python
import joblib

joblib.dump(model, "trained_model.sav")
joblib.dump(feature_extraction, "vectorizer.pkl")
```

Place `vectorizer.pkl` in this folder next to `trained_model.sav`.

The provided model is a scikit-learn Logistic Regression classifier with labels `0` and `1`. This app maps `0` to `Ham Message` and `1` to `Spam Message`.

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Deploy on Render

Use these settings:

```text
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```
