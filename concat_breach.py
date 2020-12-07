import pandas as pd

df1 = pd.read_csv('breach_report.csv')
df2 = pd.read_csv('breach_report_archive.csv')
df = df2.append(df1)
print(df)
df.to_csv('breach_report_all.csv', index=False)