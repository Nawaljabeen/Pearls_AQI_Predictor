# Pearls_AQI_Predictor by nawal 
Predicts the Air Quality Index (AQI) in your city in the next 3 days, using a 100%  serverless stack.
Streamlit app link : https://pearlsaqipredictor-aqiforecasterbynawal.streamlit.app/

Documentation : https://github.com/Nawaljabeen/Pearls_AQI_Predictor/blob/main/Lahore's%20City%20Wide%203%20day%20AQI%20Forecaster%20based%20on%20realtime%20sensor%20data.pdf

**Programming Language**

* Python 3.10

**Machine Learning & Optimization**

* **XGBoost:** Primary gradient boosting regressor used for the 24h/48h/72h base forecasts and the spatial offset model.
* **Scikit-learn:** Used for the Ridge regression baseline, TimeSeriesSplit cross-validation, and evaluation metrics (RMSE, MAE, R²).
* **Optuna:** Bayesian hyperparameter optimization engine configured for expanding-window time-series tuning.
* **Joblib:** Serializes the trained models for deployment to the registry.

**Data Engineering & MLOps**

* **Hopsworks:** Centralized MLOps platform serving as both the Feature Store (for historical AQI and weather) and the Model Registry.
* **Apache Hudi:** The underlying data format utilized by Hopsworks to enable time-travel and point-in-time correct feature retrieval.
* **Pandas & NumPy:** Core libraries for chronological data manipulation, rolling window calculations, and cyclical temporal encodings.
* **confluent-kafka:** System-level dependency for streaming data into the Hopsworks feature groups.

**Data Providers (External APIs)**

* **OpenAQ API (v3):** Source of truth for live and historical PM2.5 readings across the Clarity base station and 19 AirGradient sector stations.
* **Open-Meteo Archive API:** Source for high-resolution meteorological features (temperature, humidity, pressure, wind speed, precipitation).

**Automation & CI/CD**

* **GitHub Actions:** Orchestrates the decoupled, cron-scheduled workflows for hourly feature ingestion, hourly batch inference, and daily model retraining via Ubuntu runners.

**Frontend & Exploratory Visualization**

* **Streamlit:** Python-based web application framework for the end-user dashboard.
* **Leaflet.js / streamlit-folium:** Renders the Punjab EPA-style spatial bubble map mapping the AQI predictions to specific Lahore sectors.
* **Matplotlib & Seaborn:** Used in the EDA pipeline to quantify sensor bias, humidity drift, and feature correlation heatmaps.
