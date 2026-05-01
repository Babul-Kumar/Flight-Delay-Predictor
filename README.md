# ✈️ Flight Delay Prediction System

This project presents a **complete, end-to-end machine learning system** for predicting whether a flight will be delayed by **15 minutes or more**. It combines a **leakage-free training pipeline**, **advanced feature engineering**, and a **production-ready Streamlit web application** for real-time predictions.

The system is designed with **industry-level best practices**, ensuring reliable performance, scalability, and interpretability.

---

# 🧠 Project Overview

Flight delays are influenced by multiple factors such as airline operations, route congestion, scheduling patterns, and geographical characteristics. This project builds a robust ML solution that:

* Predicts flight delays using structured data
* Incorporates **geo-spatial and temporal features**
* Ensures **zero data leakage**
* Provides **real-time predictions via a web interface**
* Offers **explainable outputs for users**

---

# 🏗️ Key Features

## 🔒 Leakage-Free Pipeline

* All preprocessing is performed **after train-test split**
* Uses **Out-of-Fold (OOF) Target Encoding**
* Ensures no information from validation/test leaks into training

---

## 🧩 Advanced Feature Engineering

The system generates powerful features such as:

* ⏰ **Temporal Features**

  * Hour extraction
  * Cyclical encoding (sin/cos)
  * Peak-hour and night-flight indicators

* 🌍 **Geographical Features**

  * Haversine distance between airports
  * Same-state flight indicator
  * Long-haul classification

* 🛫 **Operational Features**

  * Airport traffic intensity
  * Route frequency
  * Speed (distance/time)

---

## 🤖 Machine Learning Models

The pipeline evaluates multiple models:

* Logistic Regression (baseline + interpretable)
* Polynomial Regression (captures interactions)
* AdaBoost (ensemble boosting model)

Model selection is based on:

* F1 Score (primary metric)
* ROC-AUC (ranking quality)

---

## 🎯 Threshold Optimization

Instead of using a fixed threshold (0.5), the system:

* Optimizes threshold using **Precision-Recall tradeoff**
* Improves recall for delayed flights
* Enhances real-world decision-making

---

## 🔍 Delay Cluster Profiling

* Uses **KMeans clustering**
* Identifies patterns in delay behavior
* Helps understand high-risk flight groups

---

## 🌐 Interactive Web Application

A modern Streamlit-based UI allows users to:

* Input flight details
* Get real-time predictions
* View delay probability and risk level
* Understand prediction reasoning

---

# 📂 Project Structure

```
.
├── flight_delay_assignment.py   # Training pipeline (ML + feature engineering)
├── app.py                       # Streamlit web application
├── airlines.csv                 # Airline mapping dataset
├── airports.csv                 # Airport metadata (location, city, state)
├── requirements.txt             # Dependencies
└── README.md
```

Generated locally and ignored by Git:

* `flights.csv` - training dataset
* `output/models/model.joblib` - trained model artifact
* `output/` - logs, plots, metrics, and reports

---

# ⚙️ How to Run Locally

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Configure Model Download

If `output/models/model.joblib` is not already present, set one of these before starting the app:

* `MODEL_GDRIVE_URL`
* `MODEL_GDRIVE_FILE_ID`

The app will automatically download the model into `output/models/model.joblib` on first run.

## 3️⃣ Launch the Application

```bash
streamlit run app.py
```

## 4️⃣ Access the App

Open your browser at:

```
http://localhost:8501
```

---

# 🧪 Model Inference Flow

1. User inputs flight details
2. Data is converted into a structured DataFrame
3. Preprocessing pipeline applies:

   * Feature engineering
   * Encoding
   * Feature selection
4. Model predicts probability
5. Threshold is applied to classify delay
6. Results are displayed with explanation

---

# ☁️ Deployment Options

## 🚀 1. Streamlit Community Cloud (Recommended)

* Push project to GitHub
* Visit: https://share.streamlit.io
* Create a new app
* Select repository and `app.py`
* Add `MODEL_GDRIVE_URL` or `MODEL_GDRIVE_FILE_ID` in app secrets if the model is not bundled locally
* Deploy instantly (free hosting)

---

## 🌐 2. Render

* Connect GitHub repository
* Set build command:

```bash
pip install -r requirements.txt
```

* Set start command:

```bash
streamlit run app.py --server.port $PORT
```

* Add `MODEL_GDRIVE_URL` or `MODEL_GDRIVE_FILE_ID` as an environment variable
* Deploy as a web service

---

# ⚠️ Important Notes

Keep the following files in the repository:

* `airlines.csv`
* `airports.csv`

The trained model should stay out of Git. At runtime, the app will:

* load `output/models/model.joblib` if it already exists locally
* otherwise download it from Google Drive using `gdown`

These repo files are required for:

* Feature engineering
* Model inference
* UI functionality

---

# 🏆 Key Highlights

* ✅ End-to-end ML system (data → model → UI → deployment)
* ✅ Zero data leakage architecture
* ✅ Advanced feature engineering (geo + temporal)
* ✅ Real-time prediction interface
* ✅ Explainable outputs for users
* ✅ Production-ready design

---

# 📌 Future Improvements

* Integration with real-time flight APIs
* Weather data incorporation
* Advanced models (LightGBM, XGBoost)
* SHAP-based explainability
* Mobile-friendly UI

---

# 👨‍💻 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Joblib

---

# 🎯 Conclusion

This project demonstrates how to build a **robust, real-world machine learning system** that goes beyond model training to include:

* Clean architecture
* Deployment readiness
* User interaction
* Explainability

It serves as a strong foundation for **production ML applications** and showcases practical ML engineering skills.

---
