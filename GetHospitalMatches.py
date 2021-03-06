import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

breachdf = pd.read_csv('updated_breachdf2.csv', encoding='latin1')
breachdf.rename(columns=lambda x: x.strip(), inplace=True)
#breachdf = breachdf.drop(columns=['Web Description'])
print(breachdf.isnull().sum().sum(), "null columns")

df = pd.read_csv('pos_clia_csv_sep20/pos_clia_sep20.csv', encoding='latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
     
breachdf['FAC_NAME'] = breachdf['FAC_NAME'].str.upper()      

combo = []
df['FAC_NAME'] = df['FAC_NAME'].str.upper()
for i in range(len(breachdf)):
    original = breachdf['FAC_NAME'][i]
    tmp = df[df["FAC_NAME"].str.startswith(original.split()[0])]['FAC_NAME'].values.tolist()
    if tmp:
        sim = [similar(original, val) for val in tmp]
        maxtmp = max(sim)
        combo.append(tmp[sim.index(maxtmp)])
        print(tmp[sim.index(maxtmp)])
    
print("combo created", combo[0])

arr = np.array(combo)
combo = np.hstack(arr)
print("combo", combo[0])

with open('hosp_new.txt', 'a') as writer:
    for hosp in combo:
        writer.write(hosp + "\n")
print("written to hosp_contains_new2.txt")