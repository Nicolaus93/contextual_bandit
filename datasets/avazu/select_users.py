import pandas as pd

n = 5000
data = pd.read_csv('train_data.csv')
data.drop(['C1', 'C15', 'C16', 'C18', 'C19', 'hour'], axis=1, inplace=True)
inds = data['user_id'].value_counts().index[:n]
final = data.loc[data['user_id'].isin(inds)]
final.to_csv('most_frequent_' + str(n) + '.csv', index=False)
