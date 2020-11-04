import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv('MergedBreachWithCMSDataSplit2.csv', encoding = 'latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
df = df.fillna(0)
print(df.isnull().sum(), "null columns")
print(df.columns)


from category_encoders import *

print("getting columns to encode")
columnsToEncode = list(df.select_dtypes(include=['category','object']))
encode_df = df[columnsToEncode].copy()
print("encoded df", encode_df)
 
enc = OrdinalEncoder().fit(encode_df)
print("transforming")
df_train_encoded = enc.transform(encode_df)

print("dropping")
df = df.drop(columnsToEncode, axis=1)
print("concatenating")
df = pd.concat([df, df_train_encoded])

df.to_csv("EncodedDF.csv")
print("ENCODED", df)
