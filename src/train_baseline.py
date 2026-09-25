import xgboost as xgb
import pandas as pd 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

data = pd.read_csv('data/processed/featured_data.csv')
drop_columns = ['LABEL','LABEL_5D','DATE','OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'BB_UPPER', 'BB_LOWER', 'VOLUME_SMA_10','NIFTY_CLOSE', 'NIFTY_OPEN', 'NIFTY_HIGH', 'NIFTY_LOW', 'NIFTY_SMA_20']

X = data.drop(drop_columns, axis=1)
y = data['LABEL']

split_point = int(len(data) * 0.8)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

model = xgb.XGBClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

majority_class = y_train.mode()[0]
naive_acc = (y_test == majority_class).mean()
print("\nNaive baseline (always predict majority class):", naive_acc)