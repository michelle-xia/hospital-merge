import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('MergedBreachWithCMSDataSplit2.csv', encoding = 'latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
print(df.columns)
for col in df.columns:
    values = [entry for entry in df[col]]
    values = array(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
