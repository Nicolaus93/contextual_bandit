# -*- coding: utf-8 -*-
"""
==============================
Contextual bandit on Avazu
==============================
The script uses real-world data to conduct contextual bandit experiments. Here we use
Avazu dataset, which was released by Avazu for a Kaggle competition. Please fist pre-process
datasets (use preprocess_hashing.py), and then you can run this example.
"""

import pandas as pd
import numpy as np
from bandits import thomp_cab, thomp_one, thomp_multi
import time
import os
import argparse
import pickle


def get_data(dataset):
    file_path = os.getcwd()
    d = os.path.join(os.sep, file_path, 'datasets/avazu')
    directory = os.path.join(os.sep, d, dataset)
    streaming_batch = pd.read_csv(os.path.join(os.sep, directory, 'processed.csv'))
    users = pd.read_csv(os.path.join(os.sep, directory, 'users.csv'))
    reward_list = pd.read_csv(os.path.join(os.sep, directory, 'reward_list.csv'))
    return streaming_batch, reward_list, users


def transform_data(contexts, reward_list, users, k):
    """
    """
    X = contexts.as_matrix()
    Y = reward_list.as_matrix()
    users = users.as_matrix()

    T, d = X.shape
    T = int(T/k)
    X = X.reshape((T,k,d))
    # Y = Y[:T*10]
    Y = Y.reshape((T,k))
    # users = users[:T*10]
    users = users.reshape((T,k))

    return X, Y, users


def get_data_it(dataset, K):
    """
    return iterators
    """
    file_path = os.getcwd()
    d = os.path.join(file_path, 'datasets/avazu')
    directory = os.path.join(d, dataset)
    X = pd.read_csv(os.path.join(directory,'small.csv'), chunksize=K, engine='c')
    Y = pd.read_csv(os.path.join(directory,'reward_list.csv'), chunksize=K)
    users = pd.read_csv(os.path.join(directory,'users.csv'), chunksize=K)
    return X, Y, users


# @profile
def policy_evaluation_it(policy, bandit, X, Y, users):
    """
    using iterators
    k - items per round
    THIS IS SLOW!
    """
    print(bandit)
    
    start = time.time()
    seq_error = []

    if bandit in ['Cab', 'ThompCab']:

        t = 0
        for x, y, u in zip(X,Y,users):
            if t % 100 == 0:
                print('round ' + str(t)) # debugging

            user = u.loc[0].values[0]
            full_context = x.values
            action_id = policy.get_action(full_context, user)
            reward = y.loc[action_id].values

            # update policy
            if not reward:
                policy.reward(full_context[action_id], reward, user, action_id)
                if t == 0:
                    seq_error.append(1)
                else:
                    seq_error.append(seq_error[t-1]+1)
            else:
                policy.reward(full_context[action_id], reward, user, action_id)
                if t > 0:
                    seq_error.append(seq_error[t-1])

            t += 1

    return seq_error


def policy_generation(bandit, dim, t, numUsers):
    if bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(numUsers, d=dim, gamma=0.1, delta=0.1, 
                                    R=0.02, epsilon=1/np.log(t/numUsers), random_state=3)

    elif bandit == 'ThompsonOne':
        policy = thomp_one.ThompsonOne(d=dim, delta=0.1, R=0.02, 
                                epsilon=1/np.log(t), random_state=4)

    elif bandit == 'ThompMulti':
        policy = thomp_multi.ThompMulti(numUsers, d=dim, delta=0.1, R=0.02,
                                epsilon=1/np.log(t/numUsers), random_state=4)

    elif bandit == 'random':
        policy = 0

    return policy


# @profile
def policy_evaluation(policy, bandit, X, Y, users):

    print(bandit)

    T, d, k = X.shape
    seq_error = [0] * T

    if bandit in ['ThompCab', 'ThompMulti']:
        for t in range(T):

            if t % 10000 == 0:
                print(t)

            user = users[t,0]
            full_context = X[t]
            action_id = policy.get_action(full_context, user)
            reward = Y[t, action_id]

            # update
            policy.reward(full_context[action_id], reward, action_id, user)
            if not reward:
                if t == 0:
                    seq_error[t] = 1
                else:
                    seq_error[t] = seq_error[t-1] + 1
            else:
                if t > 0:
                    seq_error[t] = seq_error[t-1]


    elif bandit in ['ThompsonOne']:
        for t in range(T):

            if t % 10000 == 0:
                print(t)

            user = users[t,0]
            full_context = X[t]
            action_id = policy.get_action(full_context)
            reward = Y[t, action_id]

            if not reward:
                policy.reward(full_context[action_id], reward, action_id)
                if t == 0:
                    seq_error[t] = 1
                else:
                    seq_error[t] = seq_error[t-1] + 1
            else:
                policy.reward(full_context[action_id], reward, action_id)
                if t > 0:
                    seq_error[t] = seq_error[t-1]

    elif bandit == 'random':
        for t in range(T):
            action_id = np.random.randint(0, k)
            reward = Y[t, action_id]
            if not reward:
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    return seq_error


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    parser = argparse.ArgumentParser(description='Multi Armed Bandit algorithms on Avazu.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to use')
    args = parser.parse_args()

    # loading dataset
    dataset = args.dataset[0]
    dataset_path = os.path.join(os.sep, os.getcwd(), 'datasets/avazu/' + dataset)
    info_file = os.path.join(os.sep, dataset_path, 'info.txt')
    for i, line in enumerate(open(info_file, 'r')):
        if i == 0:
            k = int(line.split()[0])
        elif i == 1:
            numUsers = int(line.split()[2])
        elif i == 3:
            time = int(line.split()[0]) / k
        elif i == 4:
            d = int(line.split()[0])
        print(line.rstrip())

    X, Y, users = get_data(dataset)
    X, Y, users = transform_data(X, Y, users, k)

    # create results directories
    result_dir = os.path.join(os.sep, dataset_path, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    regret_dir = os.path.join(result_dir, 'regret')
    if not os.path.exists(regret_dir):
        os.makedirs(regret_dir)
    cum_regret_dir = os.path.join(result_dir, 'cum_regret')
    if not os.path.exists(cum_regret_dir):
        os.makedirs(cum_regret_dir)

    # conduct regret analyses
    regret = {}
    cum_regret = {}

    # run algorithms
    bandits = ['ThompsonOne', 'ThompMulti', 'random']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, d, time, numUsers)
        seq_error = policy_evaluation(policy, bandit, X, Y, users)
        regret[bandit] = regret_calculation(seq_error)
        cum_regret[bandit] = seq_error
        # save results
        fileObject = open(os.path.join(cum_regret_dir, bandit), 'wb')
        regretObject = open(os.path.join(regret_dir, bandit), 'wb')
        pickle.dump(cum_regret[bandit],fileObject)
        pickle.dump(regret[bandit], regretObject)
        fileObject.close()
        regretObject.close()


if __name__ == '__main__':
    main()

