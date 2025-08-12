from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
clf = joblib.load("model/dbscan_classifier.pkl")
encoders = joblib.load("model/encoders.pkl")
scaler = joblib.load("model/scaler.pkl")
important_features = joblib.load("model/important_features.pkl")
dropdown_options = joblib.load("model/dropdown_options.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Prepare input
            input_data = {}
            for feature in important_features:
                value = request.form.get(feature)
                if feature in encoders:  # Encode categorical
                    value = encoders[feature].transform([value])[0]
                else:
                    value = float(value)
                input_data[feature] = value

            # Create DataFrame and scale
            df_input = pd.DataFrame([input_data])
            df_scaled = scaler.transform(df_input[important_features])

            # Predict
            cluster_label = clf.predict(df_scaled)[0]
            prediction = f"Cluster {cluster_label}" if cluster_label != -1 else "Noise / Outlier"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html",
                           features=important_features,
                           dropdown_options=dropdown_options,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
