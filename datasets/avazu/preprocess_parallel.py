from functools import partial
import os
import argparse
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from sklearn.preprocessing import LabelEncoder
from preprocess_hashing import feature_hashing, conjunctions
import random
import h5py


def one_hot(X):
    """
    Parallel one hot encoding
    """
    print('One-hot encoding')
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


def build_dataset(group, k):
    """
    Todo: check for duplicates between 1 and 0s
    in the process..
    """
    print('building dataset...')
    lst = []  # list containing every round
    exceptions = []
    user_interactions = group.groupby('click')  # there will be 0/1
    try:
        ones = user_interactions.get_group(1)
        # ones = ones[~ones.duplicated(keep='first')]
    except Exception as e:
        exceptions.append(e)
        # print('    not enough ones')
        return []
    try:
        zeros = user_interactions.get_group(0)
        zeros = zeros[~zeros.duplicated(keep='first')]  # discard duplicate
        for index_and_row in ones.iterrows():
            # r = pd.concat([zeros.sample(n=k - 1), index_and_row[1]
            #                .to_frame().transpose()])
            good = False
            while not good:
                r = pd.concat([zeros.sample(n=k - 1), index_and_row[1]
                              .to_frame().transpose()])
                good = is_good_round(r, k)
            lst.append(r.iloc[np.random.permutation(len(r))])  # shuffle
    except Exception as e:
        exceptions.append(e)
        # print('    not enough zeros')
        return []
    return lst


def is_good_round(rnd, k):
    r = rnd[~rnd.drop('click', axis=1).duplicated(keep='first')]
    n = r.shape[0]
    if n < k:
        return False, r
    return True, r


def build_nodupli(group, k):
    """
    Now it checks for duplicates between 1 and 0s
    """
    print('building dataset...')
    lst = []  # list containing every round
    # group by 0/1 clicks
    # user_interactions = group.drop('hour', axis=1).groupby('click')
    user_interactions = group.groupby('click')
    try:
        ones = user_interactions.get_group(1)
    except Exception:
        # this shouldn't happen but who knows :P
        print('no ones')
        return lst
    if len(ones) > 5:
        zeros = user_interactions.get_group(0)
        zeros = zeros[~zeros.duplicated(keep='first')]  # discard duplicates
        for i, index_and_row in enumerate(ones.iterrows()):
            try:
                r = pd.concat([index_and_row[1].to_frame().transpose(),
                               zeros.sample(n=k - 1)])
            except Exception:
                # print('not enough zeros')
                # there might be < 19 zeros after discarding duplicates
                print(len(zeros))
                return []
            good, r = is_good_round(r, k)
            count = 0
            while not good:
                count += 1
                r = r.append(zeros.sample(n=1))
                good, r = is_good_round(r, k)
                if count > len(ones):
                    break
            if good:
                lst.append(r.iloc[np.random.permutation(len(r))])  # shuffle
    return lst


def build_parallel(data, k, partitions):
    pool = Pool(partitions)
    dfGrouped = data.groupby(['user_id'])
    func = partial(build_nodupli, k=k)
    lst = pool.map(func, [group for name, group in dfGrouped])
    flat_list = [item for sublist in lst for item in sublist]
    # shuffle data
    random.shuffle(flat_list)
    data = pd.concat(flat_list)
    pool.close()
    pool.join()
    return data


def par_one_hot(data, partitions):
    data_split = np.array_split(data, partitions)
    pool = Pool(partitions)
    data = pd.concat(pool.map(one_hot, data_split))
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
    data_split = np.array_split(data, partitions)
    pool = Pool(partitions)
    func = partial(feature_hashing, N=n_feat)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
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
    parser.add_argument('-conj', dest='conj', action='store_true',
                        default=False,
                        help='whether include feature conjunctions or not')
    parser.add_argument(dest='name', metavar='name', type=str, nargs=1,
                        help='name of dataset after preprocessing')

    args = parser.parse_args()
    dataset = args.dataset[0]
    k = args.k[0]
    n_feat = args.n_feat[0]
    name = args.name[0]
    conj = args.conj

    # parallelize!
    cores = cpu_count()  # Number of CPU cores on your system
    partitions = cores  # Define as many partitions as you want
    print('Reading dataset')
    df = pd.read_csv(dataset)  # Load data
    df.drop('hour', axis=1, inplace=True)

    try:
        to_drop = ['C1', 'C15', 'C16', 'C18', 'C19']
        df.drop(to_drop, axis=1, inplace=True)
    except Exception:
        pass

    # building dataset
    df = build_parallel(df, k, partitions)
    print(df.shape)

    # need to reindex since we can have same row multiple times
    df = df.reset_index(drop=True)

    # conjunctions
    if conj:
        co = [c for c in df.columns if c not in ['user_id', 'click']]
        df = par_conjunctions(df, partitions, co)

    # feature hashing
    df = par_feature_hashing(df, partitions, n_feat)
    print(df.head())
    print(df.shape)

    # one-hot encoding
    df = par_one_hot(df, partitions)
    print(df.head())
    print(df.shape)

    rewards = pd.DataFrame(df['click'])
    users = pd.DataFrame(df['user_id'])
    df = df.drop(['click', 'user_id'], 1)  # remove click and user_id

    # redefine users
    le = LabelEncoder()
    le.fit(list(users['user_id']))
    users['user_id'] = le.transform(users['user_id'])
    usr_msg = str(len(users['user_id'].unique())) + \
        ' unique users after preprocessing.'

    # convert to numpy array
    t, d = df.shape
    df = df.as_matrix().reshape((t // k, k, d)).astype(np.dtype('i4'))
    rewards = rewards.as_matrix().reshape((t // k, k)).astype(np.dtype('i4'))
    users = users.as_matrix().reshape((t // k, k)).astype(np.dtype('i4'))

    # some info
    msg = 'The preprocessed dataset contains: \n  ' \
          '{} rounds \n  {} actions per round\n  ' \
          '{} columns per action\n'.format(t // k, k, d)
    print(msg)

    # save everything in hdf5 files
    print('saving...')
    file_path = os.getcwd()
    directory = os.path.join(file_path, name)
    # directory = os.path.splitext(directory)[0] + \
    #     '_k' + str(k) + '_d' + str(n_feat)
    directory = os.path.splitext(directory)[0] + '_d' + str(n_feat)
    if conj:
        directory += '_conj'
    if not os.path.exists(directory):
        os.makedirs(directory)

    data = h5py.File(os.path.join(directory, 'dataset.hdf5'), 'w')
    data.create_dataset('X', data=df, compression='gzip', compression_opts=5)
    data.create_dataset('y', data=rewards)
    data.create_dataset('users', data=users)
    data.close()

    f = open(os.path.join(directory, 'info.txt'), 'w')
    f.write(usr_msg)
    f.write('\n')
    f.write(msg)
