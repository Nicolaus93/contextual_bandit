import pandas as pd


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

    print('reading dataset...')
    df = pd.read_csv('train.csv')

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

    df.to_csv('train_data.csv', index=False)
