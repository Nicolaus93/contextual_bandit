import pandas as pd
from preprocess_hashing import conjunctions

df = pd.read_csv('train_data.csv')
to_drop = ['C1', 'hour', 'C15', 'C16', 'C18', 'C19']
df.drop(to_drop, inplace=True, axis=1)
cols = [i for i in df.columns if i not in ['click', 'user_id']]
c = conjunctions(df, cols=cols)
c.to_csv('trained_conj.csv', index=False)
