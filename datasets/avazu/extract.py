import pandas as pd
import argparse

def def_user(row):   
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']
    return user


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('-i', dest='intr', metavar='interactions', type=int, nargs=1,
                        help='number of interactions per user')

    args = parser.parse_args()
    intr = args.intr[0]
    
    df = pd.read_csv('train.csv')
    cols = ['device_id', 'device_ip', 'device_model']
    df = df.assign(user_id=pd.Series(df[cols].apply(def_user, axis=1)).values)
    n = df['user_id'].value_counts()[df['user_id'].value_counts()>=intr].index
    res = df.loc[df['user_id'].isin(n)]
    print(res.shape)
    us = len(res['user_id'].unique())
    print(us)
    name = 'filtered' + str(intr) + '.csv'
    res.to_csv(name)

