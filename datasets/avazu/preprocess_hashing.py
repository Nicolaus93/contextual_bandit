import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import random

df = pd.read_csv('filtered100.csv')
print("Unique users: " + str(len(df['device_ip'].unique())))

# Hashing features
retain = 50
y = df['click']
col = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', \
      'app_category', 'device_id', 'device_model', 'device_type', 'device_conn_type', \
      'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
enc = ce.HashingEncoder(cols=col, n_components=retain).fit(df, y)
df = enc.transform(df)

# drop something
# df['hour'] = df['hour'].apply(lambda x: str(x)[6:]) # hour
df = df.drop('hour', 1) # remove hour
df = df.drop('id', 1) # remove id

# building dataset
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
df = pd.concat(l)
rewards = pd.DataFrame(df['click'])
users = pd.DataFrame(df['device_ip'])

col = ['col_' + str(i) for i in range(retain)]
df = pd.concat([pd.get_dummies(df[c]) for c in col], axis=1) # one hot encoding
df = df.div(df.sum(axis=1), axis=0) # normalize

# redefine users
le = LabelEncoder()
le.fit(list(users['device_ip']))
users['device_ip'] = le.transform(users['device_ip'])
sorted(users['device_ip'].unique())

# save everything
rewards.to_csv('processed/filtered100/reward_list.csv')
df.to_csv('processed/filtered100/processed.csv')
users.to_csv('processed/filtered100/users.csv')