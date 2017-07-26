import pandas as pd
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select most frequent users.')
    parser.add_argument('-n', dest='n', metavar='threshold',
                        type=int, nargs=1,
                        help='number of users to include')

    args = parser.parse_args()
    n = args.n[0]
    data = pd.read_csv('train_data.csv')
    inds = data['user_id'].value_counts().index[:n]
    final = data.loc[data['user_id'].isin(inds)]
    final.to_csv('most_frequent_' + str(n) + '.csv', index=False)
