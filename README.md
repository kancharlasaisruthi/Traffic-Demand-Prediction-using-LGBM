# Traffic Demand Prediction using LightGBM

## Overview
This project predicts traffic demand using machine learning techniques and feature engineering. Multiple regression models were explored, and a LightGBM model was trained and saved for deployment.

The model achieved an R² score of ***0.9459***, demonstrating strong predictive performance on traffic demand data from Indonesia.

The application enables users to provide traffic-related inputs and obtain predicted demand values through an interactive Streamlit interface.

---

## Features

- Data preprocessing and cleaning
- Feature engineering from timestamp information
- Training and evaluation of multiple models
- LightGBM-based traffic demand prediction
- Model serialization using Pickle
- Interactive Streamlit application

---
### Feature Engineering

Several feature engineering techniques were applied to improve model performance and capture temporal traffic patterns:

- Timestamp Decomposition: Extracted time-based features such as hour, minute, day, and day of the week from the timestamp.

- Time Slot Creation: Divided the day into 15-minute intervals to represent traffic variations throughout the day.

- Rush Hour Indicator: Created a binary feature to identify peak traffic periods (7–10 AM and 5–8 PM).

- Cyclical Encoding: Applied sine and cosine transformations to hour and minute features to preserve the cyclic nature of time.

- Categorical Feature Encoding: Converted categorical variables such as road type, weather conditions, and landmarks into numerical representations suitable for machine learning models.

- Geohash-Based Location Features: Utilized geohash values representing locations in Indonesia to capture spatial information.

- Interaction Features: Combined related variables to model the influence of multiple factors on traffic demand.

- Data Cleaning and Handling Missing Values: Removed inconsistencies and prepared the dataset for training.

These engineered features enabled the LightGBM model to effectively learn temporal and spatial traffic patterns, resulting in an **R² score of 0.9459**.

---
## Dataset

The dataset contains traffic-related attributes such as:

- Geohash
- Day
- Timestamp

Target variable:

- **Demand**

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- Streamlit
- Pickle

---

## Project Structure

```
.
├── code.ipynb              # Data preprocessing, feature engineering, model training
├── streamlit_app.ipynb     # Streamlit application notebook
├── lgb_model.pkl           # Saved LightGBM model
└── README.md
```

---

## Model

The final model used is:

- **LightGBM Regressor**

Performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost streamlit
```

---

## Running the Application

Launch the Streamlit app:

```bash
streamlit run streamlit_app.ipynb
```

---
### Results
<img width="1014" height="605" alt="image" src="https://github.com/user-attachments/assets/b68d97a3-d418-4148-bd54-a72edd1e6e83" />

<img width="972" height="721" alt="image" src="https://github.com/user-attachments/assets/b84c1f85-5c1d-40e8-8053-d5195274b8f1" />

<img width="1389" height="989" alt="image" src="https://github.com/user-attachments/assets/487b9b42-195a-4edc-9c91-3afe3b0eb67d" />

<img width="1789" height="490" alt="image" src="https://github.com/user-attachments/assets/a8acae2d-8506-4731-acb4-c35e6044e7c3" />


## Future Improvements

- Hyperparameter optimization
- Ensemble learning
- Deployment on Streamlit Cloud
- Real-time traffic demand prediction
- Advanced feature engineering

---

## Author

**Sruthi Kancharla**

GitHub: https://github.com/kancharlasaisruthi
