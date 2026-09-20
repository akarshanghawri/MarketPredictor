import pandas as pd 

df = pd.read_csv('data/processed/data.csv')

# Label
# if tomorrow's close > todays close (label 1) else (label 0)
next_close = df["CLOSE"].shift(-1)
df["LABEL"] = (next_close > df["CLOSE"]).astype('Int64')

df.loc[next_close.isna(), "LABEL"] = pd.NA      # wherever shifted value is missing, make label to Nan

# Label - 5 days 
next_close_5D = df["CLOSE"].shift(-5)
df["LABEL_5D"] = (next_close_5D > df["CLOSE"]).astype('Int64')
df.loc[next_close_5D.isna(), "LABEL_5D"] = pd.NA     

# Percentage Change
prv_close = df["CLOSE"].shift(1)
df["PCT_CHANGE"] = ((df["CLOSE"] - prv_close) / prv_close * 100).round(2)

df["PCT_CHANGE_5D"] = ((df["CLOSE"] - df["CLOSE"].shift(5))/ df["CLOSE"].shift(5) * 100).round(2)

# Simple Moving Averages 
df["SMA_5"] = df["CLOSE"].rolling(window=5).mean()
df["SMA_10"] = df["CLOSE"].rolling(window=10).mean()
df["SMA_20"] = df["CLOSE"].rolling(window=20).mean()
df["SMA_50"] = df["CLOSE"].rolling(window=50).mean()

# Close to SMA
df["CLOSE_TO_SMA5"] = ((df["CLOSE"] - df["SMA_5"])/df["SMA_5"] * 100 )
df["CLOSE_TO_SMA10"] = ((df["CLOSE"] - df["SMA_10"])/df["SMA_10"] * 100 )
df["CLOSE_TO_SMA20"] = ((df["CLOSE"] - df["SMA_20"])/df["SMA_20"] * 100 )
df["CLOSE_TO_SMA50"] = ((df["CLOSE"] - df["SMA_50"])/df["SMA_50"] * 100 )

# EMA
df["EMA_12"] = df["CLOSE"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["CLOSE"].ewm(span=26, adjust=False).mean()

# MACD 
df["MACD_LINE"] = df["EMA_12"] - df["EMA_26"]
df["MACD_SIGNAL"] = df["MACD_LINE"].ewm(span=9, adjust=False).mean()
df["MACD_HIST"] = df["MACD_LINE"] - df["MACD_SIGNAL"]

# RSI
change = df["CLOSE"].diff()

gain = change.where(change > 0, 0)
loss = -change.where(change < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

RS = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + RS))

# Bollinger Bands
df["BB_UPPER"] = df["SMA_20"] + (2 * df["CLOSE"].rolling(window=20).std())
df["BB_LOWER"] = df["SMA_20"] - (2 * df["CLOSE"].rolling(window=20).std())
df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["SMA_20"]
df["BB_PCT_B"] = (df["CLOSE"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"])

# Volatility and Volume 
df["VOLATILITY_10"] = df["PCT_CHANGE"].rolling(10).std()
df["VOLATILITY_20"] = df["PCT_CHANGE"].rolling(20).std()

df["HIGH_LOW_RANGE"] = (df["HIGH"] - df["LOW"]) / df["CLOSE"]

# Volume 
prv_volume = df["VOLUME"].shift(1)
df["VOLUME_CHANGE"] = ((df["VOLUME"] - prv_volume) / prv_volume * 100)
df["VOLUME_SMA_10"] = df["VOLUME"].rolling(window=10).mean()
df["VOLUME_RATIO"] = df["VOLUME"] / df["VOLUME_SMA_10"]

# drop NaN values 
df = df.dropna()

# save the updated df 
df.to_csv("data/processed/featured_data.csv", index=False)