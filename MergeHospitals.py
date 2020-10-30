import pandas as pd
import numpy as np

with open('hosp_cont_better.txt') as f:
    cms_names = [line.rstrip() for line in f]
print("length", len(cms_names))

#cms_names.sort()


breachdf = pd.read_csv('breach_report.csv', encoding='latin1')
breachdf.rename(columns=lambda x: x.strip(), inplace=True)
breachdf = breachdf.drop(columns=['Web Description'])
#breachdf = breachdf.applymap(lambda x: x.strip() if type(x)==str else x)
#breachdf = breachdf.sort_values(by=['FAC_NAME'])
print(breachdf.isnull().sum().sum(), "null columns")

df = pd.read_csv('pos_clia_csv_sep20/pos_clia_sep20.csv', encoding='latin1', dtype=str)
df.rename(columns=lambda x: x.strip(), inplace=True)
#df = df.applymap(lambda x: x.strip() if type(x)==str else x)
#df = df.sort_values(by=['FAC_NAME'])

print('dataframes created')
breachdf['Breach Occured'] = 'No'
for i in range(len(cms_names)):
    for breachrow in breachdf['FAC_NAME']:
        #print(breachrow, cms_names[i])
        if breachrow.upper().split()[0] in cms_names[i]:
            updated_name = cms_names[i]
            print("update", updated_name)
            print("old", breachdf.loc[breachdf['FAC_NAME'] == breachrow, 'FAC_NAME'])
            breachdf.loc[breachdf['FAC_NAME'] == breachrow, 'Breach Occured'] = 'Yes'
            breachdf.loc[breachdf['FAC_NAME'] == breachrow, 'FAC_NAME'] = updated_name
            print("new", breachdf.loc[breachdf['FAC_NAME'] == breachrow, 'FAC_NAME'])
print("done", breachdf)
breachdf.to_csv('updated_breachdf2.csv', index=False)



"""df['Breach'] = "No"
df['Covered Entity Type'] = None
df['Individuals Affected'] = None
df['Breach Submission Date'] = None
df['Type of Breach'] = None
df['Location of Breached Information'] = None
df['Business Associate Present'] = None

print("breachdf")
print(breachdf)
print()
searchstr = cms_names[1].split()[0]
print("cmsnames", cms_names[0])
print("breachdf", breachdf.loc[[0]])

for i in range(len(cms_names)):
    for breachrow in breachdf['FAC_NAME']:
        #print(breachrow, cms_names[i])
        if breachrow.upper() in cms_names[i]:
            print("cmsnames", cms_names[i])
            print("index", i)
            print("breachrow", breachrow)
            print("appending", breachdf.loc[breachdf['FAC_NAME'] == breachrow])
            tmp = breachdf.loc[breachdf['FAC_NAME'] == breachrow]
            tmp = tmp.copy()
            tmp = tmp.astype('string')
            print("TMP", tmp['Covered Entity Type'])

            ind = df.index[df['FAC_NAME'] == cms_names[i]]
            df.loc[df['FAC_NAME'] == cms_names[i], 'Breach'] = 'Yes'
            a = tmp['Covered Entity Type'].astype('string')
            #df['Breach'][ind] = "Yes"
            df.loc[df['FAC_NAME'] == cms_names[i], 'Covered Entity Type'] = a #tmp['Covered Entity Type'].astype('string')
            print("DF", df['Covered Entity Type'][df['FAC_NAME'] == cms_names[i]])
            # df['Covered Entity Type'][ind] = tmp['Covered Entity Type']
            df.loc[df['FAC_NAME'] == cms_names[i], 'Individuals Affected']= tmp['Individuals Affected'].astype('|S')
            df.loc[df['FAC_NAME'] == cms_names[i], 'Breach Submission Date'] = tmp['Breach Submission Date'].astype('|S')
            df.loc[df['FAC_NAME'] == cms_names[i], 'Type of Breach'] = tmp['Type of Breach'].astype('|S')
            df.loc[df['FAC_NAME'] == cms_names[i], 'Location of Breached Information'] = tmp['Location of Breached Information'].astype('|S')
            df.loc[df['FAC_NAME'] == cms_names[i], 'Business Associate Present'] = tmp['Business Associate Present'].astype('|S')
            print("final", df['FAC_NAME'][ind])
            

            #breachdf = breachdf.drop(breachdf.loc[breachdf['FAC_NAME'] == breachrow])

#selected = breachdf.loc[breachdf['FAC_NAME'].values.str in cms_names[1]]
print("matched", df)
df.to_csv("CMSNameMatch.csv", index=False)

#mask2 = pd.Series(cms_names[1] in breachdf.FAC_NAME.values)
#mask = pd.Series(breachdf.FAC_NAME.values).str.contains(cms_names[1], case=False).values

#print('mask', mask)

#a = breachdf.loc[:,mask]

#a = breachdf.loc[breachdf['FAC_NAME'].str.contains(cms_names[0])]
#a = breachdf[breachdf['FAC_NAME'].str.startswith(cms_names[0])]"""
"""print("a", a)


df['Breach'] = "No"
df['Covered Entity Type'] = None
df['Individuals Affected'] = None
df['Breach Submission Date'] = None
df['Type of Breach'] = None
df['Location of Breached Information'] = None
df['Business Associate Present'] = None
for i in range(len(df['FAC_NAME'])):
    curr =  df['FAC_NAME'][i].upper()
    if curr in cms_names:
        ind = cms_names.index(curr)
        print("name", cms_names[i])
        mask = df[breachdf['FAC_NAME'].str.contains(cms_names[ind])]
        print(mask)
        print()
        print(df[breachdf['FAC_NAME'].str.contains(cms_names[ind])])
        df['Breach'][i] = "Yes"
        df['Covered Entity Type'][i] = mask['Covered Entity Type']
        df['Individuals Affected'][i] = mask['Individuals Affected']
        df['Breach Submission Date'][i] = mask['Breach Submission Date']
        df['Type of Breach'][i] = mask['Type of Breach']
        df['Location of Breached Information'][i] = mask['Location of Breached Information']
        df['Business Associate Present'][i] = mask['Business Associate Present']
        print(df['FAC_NAME'][i])
print("merged dataframes")
df.tocsv('MergedHospitalData.csv')"""
