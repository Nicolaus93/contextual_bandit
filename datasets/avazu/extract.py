import pandas as pd

def def_user(row):    
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']
    return user

df = pd.read_csv('train.csv')
cols = ['device_id', 'device_ip', 'device_model']
df = df.assign(user_id=pd.Series(df[cols].apply(def_user, axis=1)).values)
n = df['user_id'].value_counts()[df['user_id'].value_counts()>=100].index
res = df.loc[df['user_id'].isin(n)]
print(res.shape)
us = len(res['user_id'].unique())
print(us)
res.to_csv('filtered100.csv')