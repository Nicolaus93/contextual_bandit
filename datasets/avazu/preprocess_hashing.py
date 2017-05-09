import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import random
import datetime
import os
import argparse

def isWeekend(d):
    """
    take df['hour'] as input and check whether the day is in the weekend.
    """
    d = str(d)
    year = '20' + d[:2]
    month = d[2:4]
    day = d[4:6]
    return int(datetime.date(int(year), int(month), int(day)).isoweekday()>4)

def hour(d):
    """
    take df['hour'] as input and convert it to:
        1 (morning 6-12) or,
        2 (afternoon 13-18) or,
        3 (night 19-5)
    """
    hour = int(str(d)[6:])
    if hour >=6 and hour <=12:
        return 1
    elif hour >= 13 and hour <= 18:
        return 2
    else:
        return 3

def hashing_feat(df, n, cols):
    """
    Input: 
        - df: (pandas.DataFrame) dataset
        - n: (int) number of bins for the hashing
        - cols: (list) columns (features) to be hashed
    Output:
        - dataset: (pandas.DataFrame) hashed dataset
    """
    print("Hashing features")
    y = df['click']
    enc = ce.HashingEncoder(cols=cols, n_components=n).fit(df, y)
    return enc.transform(df)

def build_dataset(df, k):
    """
    TODO 
    """
    print("building dataset")
    grouped = df.groupby(['device_ip']) # group by users
    k = k-1 # number of 'no' clicks per round
    l = [] # list containing every round
    for name, group in grouped:
        user_interactions = group.groupby('click') # there will be 2 groups: 0/1
        try:
            zeros = user_interactions.get_group(0)
        except:
            print('    not enough zeros')
            continue
        try:
            ones = user_interactions.get_group(1)
        except:
            print('    not enough ones')
            continue
        # compute the number of splits per user and split the interactions
        num_of_splits = len(zeros)//k
        if num_of_splits > 0:
            splits = np.array_split(zeros, num_of_splits)
            # build the rounds with k-zeros rewards a 1-one reward
            for index_and_row, split in zip(ones.iterrows(), splits):
                r = pd.concat([split, index_and_row[1].to_frame().transpose()])

                if len(r)>10:
                    r.drop(r.index[:len(r)-10], inplace=True)
                # add the round to the list (after shuffling)
                l.append(r.iloc[np.random.permutation(len(r))])
        else:
            continue
    random.shuffle(l) # shuffle rounds
    return pd.concat(l)

def one_hot_enc(df):
    """
    Input: 
        - df: (pandas.DataFrame) dataset
    Output:
        normalized one hot encoded dataset (pandas.DataFrame)
    """
    print('One-hot encoding')
    col = [item for item in df.columns if item not in ['click', 'device_ip']]
    df = pd.concat([pd.get_dummies(df[c]) for c in col], axis=1) # one hot encoding
    return df.div(df.sum(axis=1), axis=0) # normalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to preprocess')
    parser.add_argument('-k', dest='k', metavar='items_per_round', type=int, nargs=1,
                        help='number of items per round')
    parser.add_argument('-d', dest='n_feat', metavar='hashed_features', type=int, nargs=1,
                        help='number of features after hashing')

    args = parser.parse_args()
    dataset = args.dataset[0]
    k = args.k[0]
    n_feat = args.n_feat[0]
    print("reading dataset")
    df = pd.read_csv(dataset)
    print("Unique users: " + str(len(df['device_id'].unique())))

    # feature engineering
    df = df.assign(day=pd.Series(df['hour'].apply(isWeekend)).values)
    df = df.assign(hour=pd.Series(df['hour'].apply(hour)).values)
    df = df.drop('id', 1) # remove id

    # Hashing features
    col = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', \
          'app_category', 'device_id', 'device_model', 'device_type', 'device_conn_type', \
          'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'hour', 'day']
    col = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', \
          'app_category', 'device_model', \
          'C14', 'C17', 'C19', 'C20', 'C21']
    df = hashing_feat(df, n_feat, col)

    # building datasets
    df = build_dataset(df, k)
    rewards = pd.DataFrame(df['click'])
    users = pd.DataFrame(df['device_ip'])

    # one hot encoding
    df = one_hot_enc(df)

    (rows, cols) = df.shape
    print("The final dataset contains: \n    -{} rows \n    -{} columns".format(rows,cols))
    # redefine users
    le = LabelEncoder()
    le.fit(list(users['device_ip']))
    users['device_ip'] = le.transform(users['device_ip'])
    sorted(users['device_ip'].unique())
    
    # save everything
    print("saving")
    file_path = os.getcwd()
    directory = os.path.join(os.sep, file_path, dataset)
    directory = os.path.splitext(directory)[0]

    if not os.path.exists(directory):
        os.makedirs(directory)
    rewards.to_csv(os.path.join(os.sep, directory, 'reward_list.csv'))
    df.to_csv(os.path.join(os.sep, directory, 'processed.csv'))
    users.to_csv(os.path.join(os.sep, directory, 'users.csv'))