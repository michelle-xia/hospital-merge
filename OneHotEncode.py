import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('TestData.csv', encoding = 'latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
#df = df.fillna(0)
print(df.isnull().sum(), "null columns")
print(df.columns)

onehot = pd.get_dummies(df)

"""#columnsToEncode = list(df.select_dtypes(include=['category','object']))
X = df.select_dtypes(include=[object])
X = X.applymap(str)
onehot = OneHotEncoder().fit_transform(X)"""
#print(onehot)
cols = onehot.columns
#for col in cols:
    #print(col)
#print(onehot['Breach Occured_Yes'])
onehot = onehot.fillna(0)
x_col_names = []
for i in range(len(onehot.columns) - 1):
    x_col_names.append(onehot.columns[i])

x = onehot[x_col_names]

y = onehot['Breach Occured_Yes']


trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=2)

model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)

"""model2 = sm.Logit(trainy, trainX)
result = model2.fit(method='newton')
print(result.summary())"""

lr_probs = model.predict_proba(testX)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(testy, lr_probs)

print('Logistic: ROC AUC=%.3f' % (lr_auc))

lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

"""#print(columnsToEncode)
#encode_df = df[columnsToEncode].copy()
le = preprocessing.LabelEncoder()
X_2 = X.apply(le.fit_transform)

enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_2)

# 3. Transform
onehotlabels = enc.transform(X_2).toarray()
onehotlabels.shape
print(onehotlabels)"""

"""for col in columnsToEncode:
    unique_vals = set()
    for val in df[col]:
        unique_vals.add(val)"""
    