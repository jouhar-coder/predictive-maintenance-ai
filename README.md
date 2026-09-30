# Predictive Maintenance AI

An end-to-end machine learning project that uses IoT sensor data to predict whether a machine will experience a catastrophic failure within the next 24 hours.

## 🚀 Live Dashboard

[Open the Predictive Maintenance AI Dashboard](https://predictive-maintenance-ai-hmalopphkmwwvmp2c4ep4q.streamlit.app/)

## 📌 Project Overview

This project demonstrates a predictive-maintenance workflow using synthetic IoT sensor telemetry.

- Synthetic sensor data generation
- Exploratory sensor analysis
- 12-hour rolling feature engineering
- 24-hour failure target creation
- Chronological 80/20 train-test split
- Random Forest classification
- Probability-threshold analysis
- Confusion matrix evaluation
- Feature-importance analysis
- Interactive Streamlit dashboard

## 🧠 Input Features

The model uses five features:

1. Sensor Temperature (°C)
2. Sensor Vibration (mm)
3. Sensor Voltage (V)
4. 12-hour Rolling Mean of Temperature
5. 12-hour Rolling Standard Deviation of Vibration

## 🤖 Model

**Algorithm:** Random Forest Classifier

Configuration used in the project:

- n_estimators = 100
- class_weight = balanced
- random_state = 42

The final analysis uses a failure-probability threshold of **0.05**.

## 📊 Final Test Results

At the selected threshold of 0.05:

| Metric | Result |
|---|---:|
| Accuracy | 96.84% |
| Precision | 13.24% |
| Recall | 69.23% |
| F1 Score | 22.22% |

Test-set confusion matrix:

```
[[1921, 59],
 [   4,  9]]
```

The test set contains 1,993 observations, including 13 actual failure cases.

## 📈 Dashboard Features

The Streamlit dashboard provides:

- Model-performance KPI cards
- Adjustable failure-probability threshold
- Current risk status
- Temperature monitoring
- Vibration monitoring
- Voltage monitoring
- Failure-probability trend
- Random Forest feature importance
- Confusion matrix
- Recent prediction results
- Maintenance recommendation

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook / Google Colab

## ⚠️ Important Limitation

This project uses **synthetic sensor data** created for demonstration and academic purposes. The model should be validated and retrained with real operational IoT data before being considered for production deployment.

## 📂 Repository Structure

```
predictive-maintenance-ai/
│
├── app.py
├── requirements.txt
└── predictive maintenance final.ipynb
```

## 👤 Author

**Jouhar M P**

GitHub: [jouhar-coder](https://github.com/jouhar-coder)