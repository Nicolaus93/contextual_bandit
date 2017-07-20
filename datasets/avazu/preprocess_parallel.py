from functools import partial
import os
import argparse
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from sklearn.preprocessing import LabelEncoder
from preprocess_hashing import feature_hashing, one_normalize, conjunctions, build_dataset


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def par_conjunctions(data, partitions, columns):
    data_split = np.array_split(data, partitions)
    pool = Pool(partitions)
    func = partial(conjunctions, cols=columns)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def par_feature_hashing(data, partitions, columns, n_feat):
    dd = data[['user_id', 'click']]
    data_split = np.array_split(data[columns], partitions)
    pool = Pool(partitions)
    func = partial(feature_hashing, N=n_feat)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    frames = [dd, data]
    data = pd.concat(frames, axis=1)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to preprocess')
    parser.add_argument('-k', dest='k', metavar='items_per_round',
                        type=int, nargs=1,
                        help='number of items per round')
    parser.add_argument('-d', dest='n_feat', metavar='hashed_features',
                        type=int, nargs=1,
                        help='number of features after hashing')

    args = parser.parse_args()
    dataset = args.dataset[0]
    k = args.k[0]
    n_feat = args.n_feat[0]

    # parallelize!
    cores = cpu_count()  # Number of CPU cores on your system
    partitions = cores  # Define as many partitions as you want
    df = pd.read_csv(dataset)  # Load data
    co = [c for c in df.columns if c not in ['user_id', 'click']]
    # df = par_conjunctions(df, partitions, co)
    df = par_feature_hashing(df, partitions, co)
    df = parallelize(df, one_normalize)
    df.to_csv('test.csv', index=False)

    # building dataset
    df = build_dataset(df, k)
    rewards = pd.DataFrame(df['click'])
    users = pd.DataFrame(df['user_id'])
    df = df.drop(['click', 'user_id'], 1)  # remove click and user_id
    usr_msg = 'There are ' + str(len(users['user_id'].unique())) + \
        ' unique users after preprocessing.'

    # redefine users
    le = LabelEncoder()
    le.fit(list(users['user_id']))
    users['user_id'] = le.transform(users['user_id'])

    # some info
    (rows, cols) = df.shape
    msg = 'The preprocessed dataset contains: \n' + \
        '  {} rows \n  {} columns.\n It looks like this:\n'.format(rows, cols)
    print(msg)
    print(df.head())

    # save everything
    print('saving...')
    file_path = os.getcwd()
    directory = os.path.join(os.sep, file_path, dataset)
    directory = os.path.splitext(directory)[0] + \
        '_k' + str(k) + '_d' + str(n_feat)

    if not os.path.exists(directory):
        os.makedirs(directory)
    rewards.to_csv(os.path.join(directory, 'reward_list.csv'), index=False)
    df.to_csv(os.path.join(directory, 'processed.csv'), index=False)
    users.to_csv(os.path.join(directory, 'users.csv'), index=False)
    f = open(os.path.join(directory, 'info.txt'), 'w')
    f.write(str(k) + ' items per round\n')
    f.write(usr_msg)
    f.write('\n')
    f.write(msg)
    f.write(df.head().to_string())
