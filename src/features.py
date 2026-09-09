import pandas as pd 
import os 

df = pd.read_csv('data/processed/data.csv')

# if yesterdays close < todays close (label 1) else (label 0)
next_close = df["CLOSE"].shift(-1)

df["LABEL"] = (next_close > df["CLOSE"]).astype('Int64')

# wherever shifted value is missing, make label to Nan
df.loc[next_close.isna(), "LABEL"] = pd.NA

df["PCT_CHANGE"] = ((next_close - df["CLOSE"]) / df["CLOSE"] * 100).round(2)

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

# drop NaN values 


# save the updated df 
df.to_csv("data/processed/data.csv", index=False)