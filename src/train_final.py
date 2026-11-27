import pandas as pd 
import xgboost as xgb
import json 
import os 
from train_baseline import get_features_and_labels

data = pd.read_csv('data/processed/featured_data.csv') 

X,y = get_features_and_labels(data)

model = xgb.XGBClassifier()
model.fit(X, y)

os.makedirs('models', exist_ok=True)
model.save_model('models/xgb_model.json')

with open('models/feature_columns.json', 'w') as f:
    json.dump(X.columns.tolist(), f)


# Verify saved model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('models/xgb_model.json')

X_sample = X.head(5)

original_predictions = model.predict(X_sample)
loaded_predictions = loaded_model.predict(X_sample)

print("Original predictions:", original_predictions)
print("Loaded predictions: ", loaded_predictions)

assert (original_predictions == loaded_predictions).all(), "Predictions do not match!"

print("Model save verification passed.")