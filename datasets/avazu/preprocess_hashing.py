import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import os
import argparse


def feature_hashing(X, N=2, cols=None):
        """A basic hashing implementation with configurable dimensionality/precision
        Performs the hashing trick on a pandas dataframe, `X`,
        using the mmh3 library.
        The number of output dimensions (`N`), and columns to hash (`cols`) are
        also configurable.
        Parameters
        ----------
        X_in: pandas dataframe
            description text
        N: int, optional
            description text
        cols: list, optional
            description text
        Returns
        -------
        out : dataframe
            A hashing encoded dataframe.
        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.
        .. [1] Kilian Weinberger; Anirban Dasgupta;
        John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing
        for Large Scale Multitask Learning. Proc. ICML.
        """
        import mmh3
        print('feature hashing...')
        if cols is None:
            cols = X.columns.values

        def hash_fn(x):
            tmp = [0 for _ in range(N)]
            for pos, val in enumerate(x.values):
                if val is not None:
                    val = str(pos) + str(val)
                    tmp[mmh3.hash(val) % N] += 1
            return pd.Series(tmp, index=new_cols)

        new_cols = ['col_%d' % d for d in range(N)]

        X_cat = X.reindex(columns=cols)
        X_num = X.reindex(columns=[x for x in X.columns.values if x not in cols])

        X_cat = X_cat.apply(hash_fn, axis=1)
        X_cat.columns = new_cols

        X = pd.merge(X_cat, X_num, left_index=True, right_index=True)

        return X


def one_hot_hashing(x):
    """
    Perform one-hot-encoding after hashing trick
    """
    x[np.where(x)[0]] = 1
    return x


def one_zero(x):
    """
    Perform one-hot encoding with pandas built-in function
    """
    x[x.nonzero()[0]] = 1
    return x


def one_normalize(x):
    """
    """
    x[np.where(x)[0]] = 1
    norm = np.sqrt(x.dot(x))
    return x / norm


def normalize(x):
    """
    So that each row has norm=1
    """
    norm = np.sqrt(x.dot(x))
    return x / norm


def build_dataset(df, k):
    """
    Todo: check for duplicates between 1 and 0s
    """
    print('building dataset...')
    grouped = df.groupby(['user_id'])   # group by users
    l = []                              # list containing every round
    for name, group in grouped:
        user_interactions = group.groupby('click')  # there will be 2 groups: 0/1
        try:
            ones = user_interactions.get_group(1)
        except:
            # print('    not enough ones')
            continue
        try:
            zeros = user_interactions.get_group(0)
            zeros = zeros[~zeros.duplicated(keep='first')]  # discard duplicates
            for index_and_row in ones.iterrows():
                r = pd.concat([zeros.sample(n=k-1), index_and_row[1].to_frame().transpose()])
                l.append(r.iloc[np.random.permutation(len(r))])
        except:
            # print('    not enough zeros')
            continue
    random.shuffle(l)
    return pd.concat(l)


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
    print('reading dataset...')
    df = pd.read_csv(dataset)

    try:
        df = df.drop('Unnamed: 0', 1)
    except:
        pass

    usr_msg = 'There are ' + str(len(df['user_id'].unique())) + ' unique users.'
    print(usr_msg)

    # Hashing features
    old_cols = df.columns  # keep these for later use
    cols = ['pub_id', 'pub_domain', 'pub_category', 'banner_pos',
            'device_model', 'device_conn_type', 'C14', 'C17',
            'C20', 'C21']
    df = feature_hashing(df, N=n_feat, cols=cols)

    # one hot encoding and normalizing
    co = [c for c in df.columns if c not in ['user_id', 'click']]
    # df[co] = df[co].apply(one_zero, axis=1)
    # df[co] = df[co].apply(normalize, axis=1)
    df[co] = df[co].apply(one_normalize, axis=1)
    print(df.head())

    # building dataset
    df = build_dataset(df, k)
    rewards = pd.DataFrame(df['click'])
    users = pd.DataFrame(df['user_id'])
    df = df.drop(['click', 'user_id'], 1)  # remove click and user_id
    usr_msg = 'There are ' + str(len(users['user_id'].unique())) + ' unique users after preprocessing.'

    # redefine users
    le = LabelEncoder()
    le.fit(list(users['user_id']))
    users['user_id'] = le.transform(users['user_id'])

    # some info
    (rows, cols) = df.shape
    msg = 'The preprocessed dataset contains: \n    {} rows \n    {} columns.\n It looks like this:\n'.format(rows,cols)
    print(msg)
    print(df.head())

    # save everything
    print('saving...')
    file_path = os.getcwd()
    directory = os.path.join(os.sep, file_path, dataset)
    directory = os.path.splitext(directory)[0] + '_k' + str(k) + '_d' + str(n_feat)

    if not os.path.exists(directory):
        os.makedirs(directory)
    rewards.to_csv(os.path.join(os.sep, directory, 'reward_list.csv'), index=False)
    df.to_csv(os.path.join(os.sep, directory, 'processed.csv'), index=False)
    users.to_csv(os.path.join(os.sep, directory, 'users.csv'), index=False)
    f = open(os.path.join(os.sep, directory, 'info.txt'), 'w')
    f.write(str(k) + ' items per round\n')
    f.write(usr_msg)
    f.write('\n')
    f.write(msg)
    f.write(df.head().to_string())
