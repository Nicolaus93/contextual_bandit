import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from preprocess_hashing import feature_hashing, one_normalize


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


cores = cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want
df = pd.read_csv('most_frequent.csv')  # Load data
co = [c for c in df.columns if c not in ['user_id', 'click']]
df[co] = df[co].apply(one_normalize, axis=1)
df = parallelize(df, feature_hashing)
df = parallelize(df, one_normalize)
df.to_csv('test.csv')
