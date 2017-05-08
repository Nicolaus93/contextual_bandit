import pandas as pd

maximum = pd.read_csv('train.csv') # [40428967 rows x 24 columns]
# select only users with more than 50 appereances
n = maximum['device_ip'].value_counts()[maximum['device_ip'].value_counts()>50].index
extracted = maximum.loc[maximum['device_ip'].isin(n)] # [18824685 rows x 24 columns]

filtered = extracted.groupby(['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'])
col = extracted.columns
for c in col:
    print(c)
    print(len(extracted[c].value_counts())