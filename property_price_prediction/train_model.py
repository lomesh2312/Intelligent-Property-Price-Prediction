import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv("data/Housing.csv")


binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "airconditioning",
    "prefarea"
]

for col in binary_columns:
    df[col] = df[col].map({"yes": 1, "no": 0})

df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

temp_rf = RandomForestRegressor(random_state=42)
temp_rf.fit(X, y)

importances = temp_rf.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("Feature Importance:\n", feature_importance_df)


top_features = feature_importance_df["Feature"].head(8).values
X = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

models = {
    "Linear Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf
}

rmse_scores = {}

for name, model in models.items():
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rmse_scores[name] = rmse

best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]

print("Best Model:", best_model_name)


joblib.dump((best_model, best_model_name, top_features),
            "model/best_house_price_model.pkl")
print("Model saved successfully!")
