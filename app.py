from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    input_text = ""
    result_class = ""

    if request.method == "POST":
        input_text = request.form.get("news_text", "")
        cleaned_text = clean_text(input_text)

        if cleaned_text:
            text_vec = vectorizer.transform([cleaned_text])
            pred = model.predict(text_vec)[0]

            proba = model.predict_proba(text_vec)[0]
            confidence = round(max(proba) * 100, 2)

            prediction = "Fake News" if pred == 1 else "Real News"
            result_class = "fake" if pred == 1 else "real"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        input_text=input_text,
        result_class=result_class
    )

if __name__ == "__main__":
    app.run(debug=True)