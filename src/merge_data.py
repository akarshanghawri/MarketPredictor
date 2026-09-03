import pandas as pd 
import os 

os.chdir('data/raw')        #change dir to where the files are stored

files = [file for file in os.listdir()]

# merge csv files
df = pd.concat(
    map(pd.read_csv, files), 
    ignore_index=True)


# sort by date
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
df = df.sort_values(by='DATE')

# columns that are needed
cols = [
    'DATE',
    'OPEN',
    'HIGH',
    'LOW',
    'CLOSE',
    'VOLUME'
]

df = df[cols]

# str -> int
numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(',', ''),
        errors='coerce'
    )

df['VOLUME'] = df['VOLUME'].astype(float)
# print(df.dtypes)

# Fix data before stock split on 28 oct 2024 (boolean masking)
ratio = 0.5024
mask = df['DATE'] < '28-10-2024'
df.loc[mask,'OPEN'] = (df.loc[mask,'OPEN'] * ratio ).round(2)
df.loc[mask,'CLOSE'] = (df.loc[mask,'CLOSE'] * ratio ).round(2)
df.loc[mask,'LOW'] = (df.loc[mask,'LOW'] * ratio ).round(2)
df.loc[mask,'HIGH'] = (df.loc[mask,'HIGH'] * ratio ).round(2)
df.loc[mask,'VOLUME'] = (df.loc[mask,'VOLUME'] / ratio ).round(0)

# save the merged file 
df.to_csv('../processed/data.csv',index=False)