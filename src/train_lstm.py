import keras
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from train_baseline import get_features_and_labels
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/processed/featured_data.csv')

X,y = get_features_and_labels(data)

top_features = [
    'MACD_LINE',
    'PCT_CHANGE_5D',
    'NIFTY_PCT_CHANGE',
    'RELATIVE_STRENGTH',
    'VOLUME_CHANGE',
    'RSI_14',
    'VOLATILITY_20',
    'BB_WIDTH',
    'CLOSE_TO_SMA20',
    'NIFTY_RSI_14',
    'HIGH_LOW_RANGE',
    'VOLUME_RATIO',
    'VOLATILITY_10',
    'NIFTY_CLOSE_TO_SMA20',
    'PCT_CHANGE'
]

X = X[top_features]

sequence_length = 10
training_ratio = 0.8

# split raw data
training_size = int(len(X) * training_ratio)

X_train_raw = X.values[:training_size]
X_test_raw = X.values[training_size:]

y_train_raw = y.values[:training_size]
y_test_raw = y.values[training_size:]

# scale using raw data
scaler = StandardScaler()

# Learn parameters from training data and transform it
X_train_scaled = scaler.fit_transform(X_train_raw)

# Apply the learned parameters to test data (do NOT fit)
X_test_scaled = scaler.transform(X_test_raw)

def generate_data(X, y, sequence_length = 10, step = 1):
    X_local = []
    y_local = []

    for start in range(0, len(X) - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(y[end-1])

    return np.array(X_local), np.array(y_local)

X_train, y_train = generate_data(X_train_scaled, y_train_raw, sequence_length)

X_test, y_test = generate_data(X_test_scaled, y_test_raw, sequence_length)

model = keras.Sequential()
model.add(LSTM(32, input_shape = (sequence_length, X.shape[1])))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy"
              , metrics=[keras.metrics.binary_accuracy]
              , optimizer="adam")

# print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.15, batch_size=64, epochs=50, callbacks=[early_stop])

model.evaluate(X_test, y_test)

y_test_prob = model.predict(X_test, verbose=1)

y_test_pred = np.where(y_test_prob > 0.5, 1, 0)

print(confusion_matrix(y_test, y_test_pred))