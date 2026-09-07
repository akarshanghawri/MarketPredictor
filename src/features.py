import pandas as pd 
import os 

df = pd.read_csv('data/processed/data.csv')

# if yesterdays close < todays close (label 1) else (label 0)
next_close = df["CLOSE"].shift(-1)

df["LABEL"] = (next_close > df["CLOSE"]).astype('Int64')

# wherever shifted value is missing, make label to Nan
df.loc[next_close.isna(), "LABEL"] = pd.NA

df["PCT_CHANGE"] = ((next_close - df["CLOSE"]) / df["CLOSE"] * 100).round(2)

df = df.dropna(subset=["LABEL"])

# save the updated df 
df.to_csv("data/processed/data.csv", index=False)