import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# inputs:
CMS_file_name = 'pos_clia_csv_sep20/pos_clia_sep20.csv'
breach_file_name = 'breach_report.csv'
matched_cms_txt = 'matching_cms_names.txt'
updated_breach_file_name = 'updated_breach.csv'
merged_file_name = 'MergedCMSWithBreaches.csv'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_matches():
    """
    This function finds the CMS hospital names which match the breached hospitals
    input: breach and CMS csv files
    output: text file with CMS hospital names which match breached hospitals
    """
    # read breach and CMS csv
    breachdf = pd.read_csv(breach_file_name, encoding='latin1')
    breachdf.rename(columns=lambda x: x.strip(), inplace=True)
    print(breachdf.isnull().sum().sum(), "null columns")

    df = pd.read_csv(CMS_file_name, encoding='latin1')
    df.rename(columns=lambda x: x.strip(), inplace=True)
        
    breachdf['FAC_NAME'] = breachdf['FAC_NAME'].str.upper()      

    combo = []
    df['FAC_NAME'] = df['FAC_NAME'].str.upper()
    for i in range(len(breachdf)):
        original = breachdf['FAC_NAME'][i]
        # ".startswith" can also be changed to ".contains" but will give less matched rows
        tmp = df[df["FAC_NAME"].str.startswith(original.split()[0])]['FAC_NAME'].values.tolist()
        # find the most similar hospital
        if tmp:
            sim = [similar(original, val) for val in tmp]
            maxtmp = max(sim)
            combo.append(tmp[sim.index(maxtmp)])
            print(tmp[sim.index(maxtmp)])
        
    print("CMS name list created", combo[0])

    # flatten list of CMS names
    arr = np.array(combo)
    combo = np.hstack(arr)
    print("combo", combo[0])

    # write CMS names to text file
    with open(matched_cms_txt, 'a') as writer:
        for hosp in combo:
            writer.write(hosp + "\n")
    print("written to " + matched_cms_txt)

def update_breaches():
    """
    This function updatesthe  breach file with matched CMS hospital names
    input: breach csv file name, CMS data file name
    output: updated_breachdf2.csv
    """
    # load matched CMS names
    with open(matched_cms_txt) as f:
        cms_names = [line.rstrip() for line in f]

    # load CMS and Breach CSV files
    breachdf = pd.read_csv(breach_file_name, encoding='latin1')
    breachdf.rename(columns=lambda x: x.strip(), inplace=True)
    breachdf = breachdf.drop(columns=['Web Description'])
    print(breachdf.isnull().sum().sum(), "null columns")
    df = pd.read_csv(CMS_file_name, encoding='latin1', dtype=str)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # create new column for whether breach occured
    breachdf['Breach Occured'] = 'No'

    # iterate through breaches and change hospital names to CMS names
    for i in range(len(cms_names)):
        for breachrow in breachdf['FAC_NAME']:
            if breachrow.upper().split()[0] in cms_names[i]:
                updated_name = cms_names[i]               
                breachdf.loc[breachdf['FAC_NAME'] == breachrow, 'Breach Occured'] = 'Yes'
                breachdf.loc[breachdf['FAC_NAME'] == breachrow, 'FAC_NAME'] = updated_name
    breachdf.to_csv(updated_breach_file_name, index=False)

    print("Written to " + updated_breach_file_name)

def merge_breached():
    """
    This function merges the CMS data with the breached data
    input: updated breach csv, CMS file
    output: merged CSV file
    """
    # read breach and CMS data
    breachdf = pd.read_csv(updated_breach_file_name, encoding='latin1')
    breachdf.rename(columns=lambda x: x.strip(), inplace=True)
    print(breachdf.isnull().sum().sum(), "null columns")

    df = pd.read_csv(CMS_file_name, encoding='latin1')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    print("dataframes read")

    # merge data
    new_df = df.merge(breachdf, left_on='FAC_NAME', right_on='FAC_NAME', how='outer')
    print("merged", new_df)

    new_df.to_csv(merged_file_name, index=False)
    print("Written to", merged_file_name)

if __name__ == "__main__":
    find_matches()
    update_breaches()
    merge_breached()