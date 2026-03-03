from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

# Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Hours_Slept": [8, 7, 7, 6, 6, 5, 5, 4],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 85]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Hours_Slept"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        study = float(request.form["study"])
        sleep = float(request.form["sleep"])
        prediction = model.predict([[study, sleep]])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)