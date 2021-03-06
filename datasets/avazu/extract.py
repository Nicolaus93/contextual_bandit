import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('-i', dest='intr', metavar='yes threshold', type=int,
                        nargs=1, help='number of yes clicks')
    parser.add_argument('-j', dest='no', metavar='no threshold', type=int,
                        nargs=1, help='number of no clicks')

    args = parser.parse_args()
    intr = args.intr[0]
    no = args.no[0]

    print('reading dataset...')
    df = pd.read_csv('train_data.csv')

    print('selecting users...')
    # select only yes clicks
    yes = df[df['click'] == 1]
    # select user whose number of yes clicks is >= than intr
    n = yes['user_id'].value_counts()[yes['user_id'].value_counts() >= intr].index
    res = df.loc[df['user_id'].isin(n)]
    # select user whose number of no clicks is >= than no
    n = res['user_id'].value_counts()[res['user_id'].value_counts() >= no].index
    res = res.loc[res['user_id'].isin(n)]
    print(res.shape)
    us = len(res['user_id'].unique())
    print(us)
    print('saving dataset...')
    name = 'filtered_' + str(intr) + 'yes_' + str(no) + 'no.csv'
    res.to_csv(name, index=False)
