import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

test_df = pd.read_excel("data/raw/test2.xlsx")

features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "sqft_living15", "sqft_lot15",
    "condition", "grade", "view", "waterfront"
]

X_test = test_df[features]

scaler = joblib.load("scaler.pkl")         
rf_model = joblib.load("tabular_rf_model.pkl")

X_test_scaled = scaler.transform(X_test)

preds_log = rf_model.predict(X_test_scaled)
preds = np.expm1(preds_log)

pred_df = pd.DataFrame({
    "id": test_df.index,
    "predicted_price": preds
})

pred_df.to_csv("predictions.csv", index=False)
print(" predictions.csv generated successfully")
