import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from preprocess_hashing import feature_hashing, one_normalize, conjunctions
from functools import partial


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def par_conj(data, partitions, columns):
   data_split = np.array_split(data, partitions)
   pool = Pool(partitions)
   func = partial(conjunctions, cols=columns)
   data = pd.concat(pool.map(func, data_split))
   pool.close()
   pool.join()
   return data


cores = cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want
dataset = 'most_frequent_10000.csv'
df = pd.read_csv(dataset)  # Load data
co = [c for c in df.columns if c not in ['user_id', 'click']]
df = par_conj(df, partitions, co)
#df = parallelize(data, conjunctions)
# df = parallelize(df, feature_hashing)
# df = parallelize(df, one_normalize)
df.to_csv('test.csv', index=False)
