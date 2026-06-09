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

## Dataset

The dataset contains traffic-related attributes such as:

- Geohash
- Day
- Timestamp
- Road Type
- Number of Lanes
- Large Vehicles
- Landmarks
- Temperature
- Weather

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
<img width="1014" height="605" alt="image" src="https://github.com/user-attachments/assets/b68d97a3-d418-4148-bd54-a72edd1e6e83" />

<img width="972" height="721" alt="image" src="https://github.com/user-attachments/assets/b84c1f85-5c1d-40e8-8053-d5195274b8f1" />

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
