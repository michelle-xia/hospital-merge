import pandas as pd

def merge_breached():

    breachdf = pd.read_csv('updated_breachdf2.csv', encoding='latin1')
    breachdf.rename(columns=lambda x: x.strip(), inplace=True)
    print(breachdf.isnull().sum().sum(), "null columns")

    df = pd.read_csv('pos_clia_csv_sep20/pos_clia_sep20.csv', encoding='latin1')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    print("dataframes read")
    new_df = df.merge(breachdf, left_on='FAC_NAME', right_on='FAC_NAME', how='outer')
    print("merged", new_df)

    new_df.to_csv("MergedBreachWithCMSDataSplit2.csv", index=False)

if __name__ == "__main__":
    merge_breached()