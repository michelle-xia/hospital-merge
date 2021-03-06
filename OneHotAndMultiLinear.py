import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pprint

df = pd.read_csv('TestData.csv', encoding = 'latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
print(df.isnull().sum(), "null columns")
print(df.columns)

pp = pprint.PrettyPrinter(indent=4)

# Encode categorical columns
onehot = pd.get_dummies(df)

cols = onehot.columns

onehot = onehot.fillna(0)

x_col_names = []

# split by X and Y
for i in range(len(onehot.columns) - 1):
    x_col_names.append(onehot.columns[i])

x = onehot[x_col_names]

y = onehot['Breach Occured_Yes']

# Split data into test/train
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=2)

# Create model
model = LogisticRegressionCV(solver='saga', max_iter=100, penalty='l1', multi_class='multinomial')
result = model.fit(trainX, trainy)

print("paramters", result.get_params(deep=True))

# predict Y values from X values
lr_probs = result.predict_proba(testX)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
lr_auc = roc_auc_score(testy, lr_probs)

print('Logistic: ROC AUC=%.3f' % (lr_auc))

# Plot false positive rate against true positive rate
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()  