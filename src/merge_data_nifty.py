import pandas as pd 
import os 

os.chdir('data/raw_nifty')        #change dir to where the files are stored

files = [file for file in os.listdir()]

# merge csv files
df = pd.concat(
    map(pd.read_csv, files), 
    ignore_index=True)

# sort by Date
df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y')
df = df.sort_values(by='Date')

# columns that are needed
cols = [
    'Date',
    'Open',
    'High',
    'Low',
    'Close'
]

df = df[cols]

# str -> int
numeric_cols = ['Open', 'High', 'Low', 'Close']
for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(',', ''),
        errors='coerce'
    )

df.to_csv('../processed/data_nifty.csv',index=False)