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
print(df.columns)
for col in df.columns:
    if df[col].dtypes == object:
        print("encoding")
        one_hot = pd.get_dummies(df[col])
        df = df.drop(col, axis = 1)
        df = df.join(one_hot)
    """values = [entry for entry in df[col]]
    values = array(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)"""
print("df", df)
# Multilinear regression
x_col_names = []
for i in range(len(df.colulmns) - 1):
    x_col_names.append(df.columns[i])

x = df[x_col_names]

y = df['Breach Occurred']

trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=2)

model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)

lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(testy, lr_probs)

print('Logistic: ROC AUC=%.3f' % (lr_auc))

lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

# pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')