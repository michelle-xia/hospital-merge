import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

df = pd.read_csv('TestData.csv', encoding = 'latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
print(df.isnull().sum(), "null columns")
print(df.columns)

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

for val in onehot['Breach Occured_Yes']:
    print(val)

# Split data into test/train
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=2)
#clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf.fit(trainX, trainy)


