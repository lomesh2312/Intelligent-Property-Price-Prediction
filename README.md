# Intelligent Property Price Prediction

A Streamlit application that predicts house prices from structured property features using a trained scikit-learn regression model.

## What It Does

- Collects property inputs (area, bedrooms, bathrooms, stories, parking, amenities, furnishing)
- Runs inference using a pre-trained best model (`best_house_price_model.pkl`)
- Shows predicted price and model name in the UI

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## Repository Structure

```text
.
├── README.md
└── property_price_prediction/
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    └── model/
        └── best_house_price_model.pkl
```

## Dataset

- Source: [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- Training script expects: `property_price_prediction/data/Housing.csv`
- Note: `data/` and CSV files are gitignored, so you must place the dataset locally before training.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
cd property_price_prediction
pip install -r requirements.txt
```

## Run the App

From `property_price_prediction/`:

```bash
streamlit run app.py
```

## Retrain the Model

1. Ensure dataset is available at `property_price_prediction/data/Housing.csv`
2. Run:

```bash
cd property_price_prediction
python train_model.py
```

This script trains three regressors:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

It selects the model with the lowest RMSE and saves:

- `property_price_prediction/model/best_house_price_model.pkl`

## Model Features Used

- `area`
- `bedrooms`
- `guestroom`
- `bathrooms`
- `mainroad`
- `prefarea`
- `stories`
- `parking`
- `basement`
- `airconditioning`
- `furnishingstatus_semi-furnished`
- `furnishingstatus_unfurnished`

## Deployment

- Streamlit Cloud app: [intelligent-property-price-prediction.streamlit.app](https://intelligent-property-price-prediction.streamlit.app/)

## Notes

- The app displays prices with INR symbol (`₹`).
- Input ranges in the UI are currently aligned to the training data assumptions hardcoded in `app.py`.
