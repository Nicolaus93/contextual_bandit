import pandas as pd
import argparse
from preprocess_hashing import conjunctions

parser = argparse.ArgumentParser(description='Get features conjunctions.')
parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                    help='the dataset to preprocess')
args = parser.parse_args()
dataset = args.dataset[0]
df = pd.read_csv(dataset)
# to_drop = ['C1', 'hour', 'C15', 'C16', 'C18', 'C19']
# df.drop(to_drop, inplace=True, axis=1)
cols = [i for i in df.columns if i not in ['click', 'user_id']]
c = conjunctions(df, cols=cols)
c.to_csv(dataset + '_conj.csv', index=False)
