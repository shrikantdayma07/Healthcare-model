import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from io import BytesIO
import base64

app = Flask(__name__, template_folder="templates")

# Model & Scaler paths
MODEL_PATH = "heart_disease_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load the trained model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
else:
    print("❌ Error: Model or scaler file not found!")
    model = None
    scaler = None

# Full list of features (same order your model expects)
ALL_FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Rest ECG", "Max Heart Rate",
    "Exercise Angina", "Oldpeak", "Slope", "CA", "Thal"
]

# Large-range features to display on the bar chart
FEATURES_FOR_CHART = [
    "Age", "Resting BP", "Cholesterol", "Max Heart Rate", "Oldpeak"
]

def generate_graph(features, values):
    """
    Creates a horizontal bar chart for selected features (skips small-range ones).
    """
    chart_features = []
    chart_values = []
    for f, v in zip(features, values):
        if f in FEATURES_FOR_CHART:
            chart_features.append(f)
            chart_values.append(v)

    plt.figure(figsize=(8, 4))
    plt.barh(chart_features, chart_values, color='skyblue')
    plt.xlabel("Values")
    plt.title("Input Feature Values")
    # Shift chart right so labels fit
    plt.subplots_adjust(left=0.28)

    # Convert plot to base64
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{graph_url}"

def generate_speedometer(prediction):
    """
    Generates a gauge chart (speedometer) with a legend,
    removing slice labels to avoid truncation.
    """
    plt.figure(figsize=(5, 5))
    colors = ["green", "red"]

    # Create the pie slices (50-50) without direct text labels
    patches, _ = plt.pie(
        [50, 50],
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.4}
    )

    # Add a legend for clarity
    plt.legend(patches, ["No Disease", "Disease Detected"], loc="best")

    # Add a label near the center
    plt.annotate("Risk Level", xy=(0, -0.1), ha="center",
                 fontsize=12, weight="bold")

    # Place an icon for the predicted slice
    if prediction == 1:  # Disease
        plt.annotate("❌", xy=(-0.2, 0.3), fontsize=20, color="red")
    else:  # No disease
        plt.annotate("✅", xy=(0.2, 0.3), fontsize=20, color="green")

    # Convert plot to base64
    img = BytesIO()
    plt.savefig(img, format="png", transparent=True)
    img.seek(0)
    speedometer_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{speedometer_url}"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "❌ Model or Scaler not found!", 500

    try:
        # Get user inputs
        feature_values = [float(request.form[field]) for field in [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]]

        # Scale input
        input_data = np.array([feature_values])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        if prediction[0] == 1:
            result_text = "High Risk! Please consult a doctor immediately."
            alert_class = "alert-danger"
            final_icon = "❌"
        else:
            result_text = "Low Risk! Keep doing exercise & yoga."
            alert_class = "alert-success"
            final_icon = "✅"

        # Generate bar chart & speedometer
        graph_url = generate_graph(ALL_FEATURE_NAMES, feature_values)
        speedometer_url = generate_speedometer(prediction[0])

        return render_template(
            "index.html",
            prediction_text=result_text,
            final_icon=final_icon,
            alert_class=alert_class,
            graph_url=graph_url,
            speedometer_url=speedometer_url
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", 400

if __name__ == '__main__':
    print("✅ Running Flask App...")
    app.run(debug=True)
