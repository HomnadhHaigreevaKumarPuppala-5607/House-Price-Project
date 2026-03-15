from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    parking = int(request.form["parking"])

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, stories, parking]],
        columns=['area','bedrooms','bathrooms','stories','parking']
    )

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ₹ {round(prediction,2)}"
    )

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=10000)
