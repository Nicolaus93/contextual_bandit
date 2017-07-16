import pandas as pd
import argparse


def def_user(row):
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']
    return user


def is_app(row):
    return True if row['site_id'] == '85f751fd' else False


def def_pub(x):
    y = {}
    if is_app(x):
        y['pub_id'] = x['app_id']
        y['pub_domain'] = x['app_domain']
        y['pub_category'] = x['app_category']
    else:
        y['pub_id'] = x['site_id']
        y['pub_domain'] = x['site_domain']
        y['pub_category'] = x['site_category']
    return pd.Series([y['pub_id'], y['pub_domain'], y['pub_category']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('-i', dest='intr', metavar='yes threshold', type=int, nargs=1,
                        help='number of yes clicks')
    parser.add_argument('-j', dest='no', metavar='no threshold', type=int, nargs=1,
                        help='number of no clicks')

    args = parser.parse_args()
    intr = args.intr[0]
    no = args.no[0]

    print('reading dataset...')
    df = pd.read_csv('train_' + str(intr))
    # introduce user_id
    print('defining user...')
    cols = ['device_id', 'device_ip', 'device_model']
    df = df.assign(user_id=pd.Series(df[cols].apply(def_user, axis=1)).values)

    print('defining columns')
    cols = ['app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']
    newcols = df[cols].apply(def_pub, axis=1)
    newcols.columns = ['pub_id', 'pub_domain', 'pub_category']
    df = df.join(newcols)
    del newcols
    # remove unnecessary features
    cols.extend(('id', 'device_id', 'device_ip', 'device_type'))
    df = df.drop(cols, 1)  # remove id

    df.to_csv('train_' + str(intr) + 'users.csv', index=False)

    # print('selecting users...')
    # # select only yes clicks
    # yes = df[df['click']==1]
    # # select user whose number of yes clicks is >= than intr 
    # n = yes['user_id'].value_counts()[yes['user_id'].value_counts()>=intr].index
    # res = df.loc[df['user_id'].isin(n)]
    # # select user whose number of no clicks is >= than no 
    # n = res['user_id'].value_counts()[res['user_id'].value_counts()>=no].index
    # res = res.loc[res['user_id'].isin(n)]
    # print(res.shape)
    # us = len(res['user_id'].unique())
    # print(us)
    # print('saving dataset...')
    # name = 'filtered_' + str(intr) + 'yes_' + str(no) + 'no.csv'
    # res.to_csv(name, index=False)
