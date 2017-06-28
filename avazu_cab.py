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
from multiprocess import thomp_cab
# from striatum.bandit import linthompsamp, linucb, cab, thomp_cab
import time
import os
import argparse
import pickle


def get_data(dataset):
    file_path = os.getcwd()
    d = os.path.join(os.sep, file_path, 'datasets/avazu')
    directory = os.path.join(os.sep, d, dataset)
    streaming_batch = pd.read_csv(os.path.join(os.sep, directory, 'processed.csv'))
    try:
        streaming_batch.drop('Unnamed: 0', inplace=True, axis=1)
    except:
        pass
    users = pd.read_csv(os.path.join(os.sep, directory, 'users.csv'))
    reward_list = pd.read_csv(os.path.join(os.sep, directory, 'reward_list.csv'))
    return streaming_batch, users, reward_list


def policy_generation(bandit, dim, t, numUsers):
    if bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(numUsers, d=dim, minUsed=1, p=1,
                                        gamma=0.2, delta=0.1, R=0.02, epsilon=1/np.log(t))
    elif bandit == 'random':
        policy = 0

    return policy


def policy_evaluation(policy, bandit, streaming_batch, users, reward_list, k):
    print(bandit)
    times = len(streaming_batch) // k
    seq_error = np.zeros(shape=(times, 1))
    action_ids = range(k)
    start = time.time()

    if bandit in ['LinUCB', 'LinThompSamp']:
        for t in range(times):
            full_context = {}
            for action_id in action_ids:
                full_context[action_id] = np.array(streaming_batch.iloc[t*k+action_id][1:]) # don't include the index

            # get next (one) action to perform and its reward
            history_id, action = policy.get_action(full_context, 1)
            reward = reward_list.iloc[t*k+action[0].action]['click']
            # update policy
            if not reward:
                policy.reward(history_id, {action[0].action: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                policy.reward(history_id, {action[0].action: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit in ['Cab', 'ThompCab']:
        for t in range(times):
            if t%100==0:
                print('round ' + str(t)) # debugging

            user = users.iloc[t*k]['user_id']
            full_context = streaming_batch.iloc[t*k:t*k+10].values
            action_id = policy.get_action(full_context, user)
            reward = reward_list.iloc[t*k+action_id]['click']

            # update policy
            if not reward:
                policy.reward(full_context[action_id], reward, user, action_id)

                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                policy.reward(full_context[action_id], reward, user, action_id)
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit == 'random':
        for t in range(times):
            action = np.random.randint(0, 10)
            reward = reward_list.iloc[t*k+action]['click']
            if not reward:
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    end = time.time()
    print('time: {} sec'.format(end-start))

    return seq_error


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    parser = argparse.ArgumentParser(description='Multi Armed Bandit algorithms.')
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
        print(line.rstrip())
    streaming_batch, users, reward_list = get_data(dataset)
    numUsers = len(users['user_id'].unique())

    time = len(streaming_batch)//k
    d = streaming_batch.shape[1]
    print("rounds: {}".format(time))

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
    bandits = ['ThompCab', 'random']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, d, time, numUsers)
        seq_error = policy_evaluation(policy, bandit, streaming_batch, users, reward_list, k)
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

