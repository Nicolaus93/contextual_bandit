import pandas as pd
import argparse
from multiprocessing import cpu_count, Pool
from functools import partial


def select(data, k, partitions):
    pool = Pool(partitions)
    dfGrouped = data.groupby(['user_id'])
    func = partial(select_users, k=k)
    lst = pool.map(func, [group for name, group in dfGrouped])
    data = pd.concat(lst)
    pool.close()
    pool.join()
    return data


def select_users(group, k):
    """
    For parallel usage
    """
    user_interactions = group.groupby('click')  # there will be 0/1
    try:
        ones = user_interactions.get_group(1)
    except Exception:
        return pd.DataFrame()
    try:
        zeros = user_interactions.get_group(0)
    except Exception:
        return pd.DataFrame()
    ratio = len(ones) / (len(zeros) + len(ones))
    if ratio > k:
        return pd.concat([ones, zeros])
    return pd.DataFrame()


def select_seq(df, k):
    """
    Sequential
    """
    lst = []
    grouped = df.groupby(['user_id'])   # group by users
    i = 0
    for name, group in grouped:
        i += 1
        user_interactions = group.groupby('click')  # there will be 0/1
        try:
            ones = user_interactions.get_group(1)
        except Exception:
            continue
        try:
            zeros = user_interactions.get_group(0)
        except Exception:
            continue
        ratio = len(ones) / (len(zeros) + len(ones))
        if ratio > k:
            lst.append(pd.concat([ones, zeros]))
    return pd.concat(lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select users.')
    # parser.add_argument('-i', dest='sup', metavar='upper', type=float,
    #                     nargs=1, help='upper threshold')
    # parser.add_argument('-j', dest='inf', metavar='lower', type=float,
    #                     nargs=1, help='lower threshold')
    parser.add_argument('-i', dest='threshold', type=float, nargs=1,
                        help='ratio between yes and no clicks')

    args = parser.parse_args()
    # sup = args.sup[0]
    # inf = args.inf[0]
    i = args.threshold[0]

    # partitions = cpu_count()
    partitions = 24
    print('reading dataset...')
    df = pd.read_csv('preprocessed.csv')

    print('selecting users...')
    # df = select(df, i, partitions)
    df = select_seq(df, i)

    print(df.shape)
    us = len(df['user_id'].unique())
    print('There are {} users'.format(len(df['user_id'].unique())))
    print('saving dataset...')
    name = 'preprocessed_' + str(i) + '.csv'
    df.to_csv(name, index=False)
