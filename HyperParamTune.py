import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

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

data = [x, y]
print("*"* 5 , "data")
print(data[0])
print(data[1])
#models = {'OLS': linear_model.LinearRegression(),
         #'Lasso': linear_model.Lasso(),
         #'Ridge': linear_model.Ridge(),}
lasso_params = {'alpha':[0.02, 0.024, 0.03]}
ridge_params = {'alpha':[200, 300, 500]}

models2 = {'OLS': linear_model.LinearRegression(),
    'Lasso': GridSearchCV(cv = 3, estimator=linear_model.Lasso(), 
                        param_grid=lasso_params).fit(data[0], data[1]).best_estimator_,
    'Ridge': GridSearchCV(cv = 3, estimator=linear_model.Ridge(), 
                        param_grid=ridge_params).fit(data[0], data[1]).best_estimator_,}

def test(models, data, iterations = 5):
    print("entering test")
    results = {}
    for i in models:
        print("starting iteration", i)
        r2_train = []
        r2_test = []
        for j in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(data[0], 
                                                                data[1], 
                                                             test_size= 0.2)
            print("r2 test") 
            r2_test.append(metrics.r2_score(y_test,
                                            models[i].fit(X_train, 
                                                         y_train).predict(X_test)))
            print("r2 train") 
            r2_train.append(metrics.r2_score(y_train, 
                                             models[i].fit(X_train, 
                                                          y_train).predict(X_train)))
            print("iteration", j)
        results[i] = [np.mean(r2_train), np.mean(r2_test)]
    return pd.DataFrame(results)

    

    

#if __name__ == "__main__":
    """for j in range(5):
        X_train, X_test, y_train, y_test = train_test_split(data[0], 
                                                                    data[1], 
                                                                test_size= 0.2)
        lr_probs = linear_model.LinearRegression().fit(X_train, y_train).predict(X_test)
        lr_auc = roc_auc_score(y_test, lr_probs)
        print("AUC", lr_auc)"""
print(test(models2, data))