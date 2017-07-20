from functools import partial
import os
import argparse
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from sklearn.preprocessing import LabelEncoder
from preprocess_hashing import feature_hashing, one_normalize, conjunctions, build_dataset


def one_hot(X):
    """
    """
    print("One-hot encoding")
    cols = [i for i in X.columns.values if i not in ['user_id', 'click']]

    def one_hot_encoding(x):
        """
        Perform one-hot encoding and normalize
        """
        x[np.where(x)[0]] = 1
        return x

    X_cat = X.reindex(columns=cols)
    X_num = X.reindex(columns=[x for x in X.columns.values
                               if x not in cols])
    X_cat = X_cat.apply(one_hot_encoding, axis=1)
    X = pd.merge(X_cat, X_num, left_index=True, right_index=True)
    return X


# def applyParallel(dfGrouped, func):
#     with Pool(cpu_count()) as p:
#         ret_list = p.map(func, [group for name, group in dfGrouped])
#     return pd.concat(ret_list)

#     # applyParallel(df.groupby(df.index), tmpFunc)


# def build_parallel(dfGrouped, func):
#     pool = Pool(partitions)
#     data = pd.concat(pool.map(build_dataset, [group for name, group in dfGrouped]))
#     pool.close()
#     pool.join()
#     return data


def par_one_hot(data, partitions):
    data_split = np.array_split(data, partitions)
    pool = Pool(partitions)
    data = pd.concat(pool.map(one_hot, data_split))
    pool.close()
    pool.join()
    return data


def parallelize(data, func, partitions):
    data_split = np.array_split(data, partitions)
    pool = Pool(partitions)
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


def par_feature_hashing(data, partitions, n_feat=60):
    # dd = data[['user_id', 'click']]
    # data_split = np.array_split(data[columns], partitions)
    data_split = np.array_split(data, partitions)
    pool = Pool(partitions)
    func = partial(feature_hashing, N=n_feat)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    # frames = [dd, data]
    # data = pd.concat(frames, axis=1)
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
    print('Reading dataset')
    df = pd.read_csv(dataset)  # Load data
    co = [c for c in df.columns if c not in ['user_id', 'click']]
    df = par_conjunctions(df, partitions, co)
    df = par_feature_hashing(df, partitions, n_feat)
    print(df.head())
    # df = parallelize(df, one_hot_encoding, partitions)
    df = par_one_hot(df, partitions)
    print(df.head())

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
