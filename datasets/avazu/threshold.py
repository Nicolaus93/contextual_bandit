import pandas as pd
import argparse
from multiprocessing import cpu_count, Pool
from functools import partial


def select(data, k, partitions):
    pool = Pool(partitions)
    dfGrouped = data.groupby(['user_id'])
    func = partial(select_users, k=k)
    lst = pool.map(func, [group for name, group in dfGrouped])
    flat_list = [item for sublist in lst for item in sublist]
    data = pd.concat(flat_list)
    pool.close()
    pool.join()
    return data


def select_users(group, k):
    """
    """
    lst = []  # list containing every round
    exceptions = []
    user_interactions = group.groupby('click')  # there will be 0/1
    try:
        ones = user_interactions.get_group(1)
        # ones = ones[~ones.duplicated(keep='first')]
    except Exception as e:
        exceptions.append(e)
        return []
    try:
        zeros = user_interactions.get_group(0)
        # zeros = zeros[~zeros.duplicated(keep='first')]  # discard duplicate
    except Exception as e:
        exceptions.append(e)
        return []
    ratio = len(ones) / len(zeros)
    if ratio > k:
        lst.append(user_interactions)
    return lst


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

    partitions = cpu_count()
    print('reading dataset...')
    df = pd.read_csv('train_data.csv')

    print('selecting users...')
    df = select(df, i, partitions)

    print(df.shape)
    us = len(df['user_id'].unique())
    print('There are {} users'.format(len(df['user_id'].unique())))
    print('saving dataset...')
    name = 'filtered_' + str(i) + '.csv'
    df.to_csv(name, index=False)
