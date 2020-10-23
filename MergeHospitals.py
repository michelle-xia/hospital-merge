import pandas as pd

with open('hosp_contain.txt') as f:
    cms_names = [line.rstrip() for line in f]

cms_names.sort()
print(cms_names[0])

breachdf = pd.read_csv('breach_report.csv', encoding='latin1')
breachdf.rename(columns=lambda x: x.strip(), inplace=True)
breachdf = breachdf.drop(columns=['Web Description'])
breachdf = breachdf.applymap(lambda x: x.strip() if type(x)==str else x)
breachdf = breachdf.sort_values(by=['FAC_NAME'])
print(breachdf.isnull().sum().sum(), "null columns")

df = pd.read_csv('pos_clia_csv_sep20/pos_clia_sep20.csv', encoding='latin1')
df.rename(columns=lambda x: x.strip(), inplace=True)
df = df.applymap(lambda x: x.strip() if type(x)==str else x)
df = df.sort_values(by=['FAC_NAME'])
print('dataframes sorted')


