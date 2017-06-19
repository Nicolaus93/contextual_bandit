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
import matplotlib.pyplot as plt
from striatum.storage import history
from striatum.storage import model
from striatum.storage import action
from striatum.bandit import linthompsamp, linucb, cab, thomp_cab
import time
import os
import argparse

aa = 1

def get_data(dataset):
    file_path = os.getcwd()
    d = os.path.join(os.sep, file_path, 'datasets/avazu')
    directory = os.path.join(os.sep, d, dataset)
    streaming_batch = pd.read_csv(os.path.join(os.sep, directory, 'processed.csv'))
    users = pd.read_csv(os.path.join(os.sep, directory, 'users.csv'))
    reward_list = pd.read_csv(os.path.join(os.sep, directory, 'reward_list.csv'))
    return streaming_batch, users, reward_list


def policy_generation(bandit, dim, k):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    actionstorage = list(range(k))

    if bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(historystorage, modelstorage, actionstorage,
                                           context_dimension=dim, delta=0.1, R=0.01, epsilon=1/np.log(k))

    elif bandit == 'LinUCB':
        policy = linucb.LinUCB(historystorage, modelstorage, actionstorage, alpha=0.3, context_dimension=dim)

    elif bandit == 'Cab':
        policy = cab.CAB(historystorage, modelstorage, actionstorage, 6040, context_dimension=dim, minUsed=1)

    elif bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(historystorage, modelstorage, actionstorage, 6040, context_dimension=dim, minUsed=1, p=0.2,
                                        gamma=0.2, delta=0.1, R=0.01, epsilon=1/np.log(k))

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
            user = users.iloc[t*k]['device_ip']
            full_context = {}
            for action_id in action_ids:
                full_context[action_id] = np.array(streaming_batch.iloc[t*k+action_id][1:])

            history_id, action = policy.get_action(full_context, user)
            reward = reward_list.iloc[t*k+action[0].action]['click']
            # update policy
            if not reward:
                policy.reward(history_id, {action[0].action: 0.0}, user)
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                policy.reward(history_id, {action[0].action: 1.0}, user)
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
    dataset = args.dataset[0]
    info_path = 'datasets/avazu/' + dataset + '/info.txt'
    info_file = os.path.join(os.sep, os.getcwd(), info_path)
    for i, line in enumerate(open(info_file, 'r')):
        if i == 0:
            k = int(line.split()[0])
        print(line.rstrip())
    streaming_batch, users, reward_list = get_data(dataset)
    streaming_batch = streaming_batch.iloc[:20000]
    time = len(streaming_batch)//k
    d = streaming_batch.shape[1]-1
    print("rounds: {}".format(time))
    # conduct regret analyses
    regret = {}
    cum_regret = {}
    col = ['b', 'g', 'r', 'y']
    # bandits = ['Cab', 'ThompCab', 'LinThompSamp', 'random']
    bandits = ['ThompCab', 'LinThompSamp', 'random']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, d, k)
        seq_error = policy_evaluation(policy, bandit, streaming_batch, users, reward_list, k)
        regret[bandit] = regret_calculation(seq_error)
        cum_regret[bandit] = seq_error
        plt.plot(range(time), cum_regret[bandit], c=col[i], ls='-', label=bandit)
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend(loc='upper left')
        axes = plt.gca()
        plt.title("Regret Bound with respect to T")
    plt.show()


if __name__ == '__main__':
    main()

