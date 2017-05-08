import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

df = pd.read_csv('avazu_medium.csv')
le = LabelEncoder()
for col in ['site_id','site_domain','site_category','app_id','app_domain',
            'app_category','device_model','device_id','device_ip']:
    le.fit(list(df[col]))
    df[col] = le.transform(df[col])

# filtering users
n = df['device_ip'].value_counts()[df['device_ip'].value_counts()>=50].index
df = df.loc[df['device_ip'].isin(n)]
# modifying features
df['hour'] = df['hour'].apply(lambda x: str(x)[6:]) # modify hour
df = df.drop('id', 1) # remove id

grouped = df.groupby(['device_ip']) # group by users
k = 9 # number of no click per round
l = [] # list containing every round
for name, group in grouped:
    user_interactions = group.groupby('click') # there will be 2 groups: 0/1
    try:
        ones = user_interactions.get_group(1)
    except:
        print('no ones')
        continue
    zeros = user_interactions.get_group(0)
    # compute the number of splits per user and split the interactions
    num_of_splits = len(zeros)//k
    splits = np.array_split(zeros, num_of_splits)
    # build the rounds with k-zeros rewards a 1-one reward
    for index_and_row, split in zip(ones.iterrows(), splits):
        r = pd.concat([split, index_and_row[1].to_frame().transpose()])
        # add the round to the list (after shuffling)
        if len(r)>10:
            r.drop(r.index[:len(r)-10], inplace=True)
        l.append(r.iloc[np.random.permutation(len(r))])

random.shuffle(l)
processed = pd.concat(l)
# define rewards
rewards = pd.DataFrame(processed['click'])
rewards.to_csv('processed/reward_list.csv')
# build the final dataset
col = ['C1', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type', 'C15', 'C16', 'C18']
final = pd.concat([pd.get_dummies(processed[c]) for c in col], axis=1)
final = final.div(final.sum(axis=1), axis=0) # normalize
final.to_csv('processed/10k_medium.csv')
# define users
users = pd.DataFrame(processed['device_ip'])
# reassign id to users
le.fit(list(users['device_ip']))
users['device_ip'] = le.transform(users['device_ip'])
users.to_csv('processed/users.csv')