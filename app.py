import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .section-note {
        color: #64748b;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">⚙️ Predictive Maintenance AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Machine failure prediction using IoT sensor data • Random Forest</div>',
    unsafe_allow_html=True,
)

st.info(
    "This dashboard reproduces the predictive-maintenance workflow "
    "from the project notebook using synthetic IoT sensor data."
)

# ---------------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------------
np.random.seed(42)

timestamps = pd.date_range(
    start="2025-01-01",
    periods=10000,
    freq="h",
)

temperature = np.random.normal(70, 4, 10000)
vibration = np.random.normal(2.5, 0.3, 10000)
voltage = np.random.normal(220, 4, 10000)

failure = np.zeros(10000, dtype=int)

failure_times = np.arange(500, 10000, 150)

for f in failure_times:
    start = f - 48

    for i in range(start, f):
        progress = (i - start) / 48
        temperature[i] += progress * 18
        vibration[i] += progress * 1.5
        voltage[i] -= progress * 12

    temperature[f] += 20
    vibration[f] += 2
    voltage[f] -= 15
    failure[f] = 1

df = pd.DataFrame(
    {
        "Timestamp": timestamps,
        "Sensor Temp C": temperature,
        "Sensor Vibration mm": vibration,
        "Sensor Voltage V": voltage,
        "Catastrophic Failure": failure,
    }
)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
df = df.sort_values("Timestamp")

df["Temp_Rolling_Mean_12h"] = (
    df["Sensor Temp C"].rolling(window=12).mean()
)

df["Vibration_Rolling_Std_12h"] = (
    df["Sensor Vibration mm"].rolling(window=12).std()
)

df = df.dropna().reset_index(drop=True)

df["Failure_In_24_Hours"] = (
    df["Catastrophic Failure"].shift(-24)
)

df = df.dropna(subset=["Failure_In_24_Hours"]).copy()

df["Failure_In_24_Hours"] = (
    df["Failure_In_24_Hours"].astype(int)
)

# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------
features = [
    "Sensor Temp C",
    "Sensor Vibration mm",
    "Sensor Voltage V",
    "Temp_Rolling_Mean_12h",
    "Vibration_Rolling_Std_12h",
]

X = df[features]
y = df["Failure_In_24_Hours"]

# ---------------------------------------------------------
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ---------------------------------------------------------
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ---------------------------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Prediction Settings")

threshold = st.sidebar.slider(
    "Failure Probability Threshold",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project")
st.sidebar.caption("Predictive Maintenance AI")
st.sidebar.caption("Random Forest • IoT Sensor Analytics")
st.sidebar.caption("Synthetic dataset")

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (
    y_prob >= threshold
).astype(int)

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

# ---------------------------------------------------------
# TOP KPI CARDS
# ---------------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Precision", f"{precision:.2%}")

with col3:
    st.metric("Recall", f"{recall:.2%}")

with col4:
    st.metric("F1 Score", f"{f1:.2%}")

st.markdown(
    '<div class="section-note">Metrics are calculated on the chronological test set at the selected probability threshold.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# CURRENT RISK STATUS
# ---------------------------------------------------------
latest_probability = float(y_prob[-1])
latest_timestamp = df.iloc[split_index + len(y_prob) - 1]["Timestamp"]

st.subheader("🚨 Current Risk Status")

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    if latest_probability >= threshold:
        st.error("🔴 FAILURE RISK DETECTED")
    else:
        st.success("🟢 NO FAILURE ALERT")

with risk_col2:
    st.metric(
        "Latest Failure Probability",
        f"{latest_probability:.2%}",
    )

with risk_col3:
    st.metric(
        "Alert Threshold",
        f"{threshold:.2%}",
    )

st.caption(
    f"Latest test observation: {latest_timestamp:%Y-%m-%d %H:%M}"
)

# ---------------------------------------------------------
# SENSOR OVERVIEW
# ---------------------------------------------------------
st.subheader("📈 Sensor Overview")

# Downsample only for visualization so the charts remain readable.
chart_df = df.iloc[::10].copy()

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(
        chart_df["Timestamp"],
        chart_df["Sensor Temp C"],
        linewidth=1.2,
    )
    ax.set_title("Machine Temperature Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(
        chart_df["Timestamp"],
        chart_df["Sensor Vibration mm"],
        linewidth=1.2,
    )
    ax.set_title("Machine Vibration Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Vibration (mm)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# VOLTAGE
# ---------------------------------------------------------
st.subheader("⚡ Voltage Monitoring")

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(
    chart_df["Timestamp"],
    chart_df["Sensor Voltage V"],
    linewidth=1.2,
)
ax.set_title("Machine Voltage Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Voltage (V)")
ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# FAILURE PROBABILITY
# ---------------------------------------------------------
st.subheader("🎯 Predicted Failure Probability")

probability_df = pd.DataFrame(
    {
        "Timestamp": df.iloc[split_index:]["Timestamp"].values,
        "Failure Probability": y_prob,
    }
)

prob_chart_df = probability_df.iloc[::5].copy()

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(
    prob_chart_df["Timestamp"],
    prob_chart_df["Failure Probability"],
    linewidth=1.2,
    label="Predicted probability",
)
ax.axhline(
    threshold,
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold = {threshold:.2f}",
)
ax.set_title("Predicted Failure Probability — Next 24 Hours")
ax.set_xlabel("Time")
ax.set_ylabel("Probability")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# FEATURE IMPORTANCE + CONFUSION MATRIX
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🔍 Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Failure", "Failure"])
    ax.set_yticklabels(["No Failure", "Failure"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# PREDICTION RESULTS
# ---------------------------------------------------------
st.subheader("📋 Recent Prediction Results")

results = X_test.copy()

results["Actual Failure"] = y_test.values
results["Failure Probability"] = y_prob
results["Predicted Failure"] = y_pred

display_results = results.tail(100).copy()

st.dataframe(
    display_results,
    width="stretch",
    height=360,
)

# ---------------------------------------------------------
# BUSINESS RECOMMENDATION
# ---------------------------------------------------------
st.subheader("💡 Maintenance Recommendation")

st.write(
    "Use the model as an early-warning system for potential "
    "machine failure within the next 24 hours. The 12-hour "
    "rolling mean of temperature is particularly important "
    "in this project. Model alerts should be combined with "
    "sensor inspection and maintenance-team verification "
    "before taking maintenance action."
)

st.warning(
    "⚠️ This project uses synthetic sensor data. "
    "The model should be validated with real operational "
    "IoT data before deployment."
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "Predictive Maintenance AI | Random Forest | "
    "IoT Sensor Analytics"
)