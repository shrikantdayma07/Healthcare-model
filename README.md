**Heart Disease Prediction**

This repository contains a Heart Disease Prediction project that uses Machine Learning (Random Forest) to estimate a person’s risk of heart disease. A Flask web interface provides an intuitive way to input health parameters and visualize the prediction results.

**1. Overview**
Goal: Predict whether a person is at risk of heart disease based on medical attributes (e.g., age, cholesterol).
Approach: A Random Forest Classifier trained on a Kaggle-based heart disease dataset.
Accuracy: Achieves around 85% on test data.
UI: A Flask web app with bar chart and speedometer visualizations to illustrate the input values and final risk level.

**2. Dataset**
We used a public heart disease dataset from Kaggle. The dataset includes the following key features:

Age: Age in years
Sex: (1 = Male, 0 = Female)
Chest Pain Type (0–3)
Resting Blood Pressure (mm Hg)
Cholesterol (mg/dl)
Fasting Blood Sugar (> 120 mg/dl: 1 = True, 0 = False)
Rest ECG (0–2)
Max Heart Rate (Thalach)
Exercise-Induced Angina (1 = Yes, 0 = No)
Oldpeak (ST Depression)
Slope (0–2)
CA: Number of major vessels (0–4)
Thal (0–3)
The target variable is whether heart disease is present (1) or not (0).

**3. Technologies Used**
Python 3 for all data processing and model building.
scikit-learn for the Random Forest Classifier and data preprocessing (StandardScaler).
Flask to create a web application that collects user inputs and displays predictions.
Matplotlib to generate bar charts (for user input features) and a speedometer gauge (risk indicator).
HTML/CSS/Bootstrap for a responsive and user-friendly UI.

**4. How It Works**

**I] Data Preprocessing:**
We clean the dataset and scale input features using StandardScaler.

**II]Model Training:**
A Random Forest Classifier is trained to classify heart disease risk (0 or 1).
Achieves around 85% accuracy on the test set.

**III]Flask App:**
Users enter their health details (age, blood pressure, etc.) in a web form.
The app scales these inputs and runs them through the trained model.
Visualizations:
A bar chart compares the user’s input features (like cholesterol) with the normal range.
A speedometer gauge indicates “No Disease” vs. “Disease Detected.”

**IV]Result:**
A color-coded message displays “High Risk” or “Low Risk”, along with recommended actions (consult a doctor, maintain exercise, etc.).
