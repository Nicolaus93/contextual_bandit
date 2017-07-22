# -*- coding: utf-8 -*-
"""
==============================
Contextual bandit on Avazu
==============================
The script uses real-world data to conduct contextual bandit experiments.
Here we use Avazu dataset, which was released by Avazu for a Kaggle
competition. Please fist pre-process datasets (use preprocess_hashing.py),
and then you can run this example.
"""

import numpy as np
import time
import os
import argparse
import pickle
import h5py
from bandits import thomp_cab, thomp_one, thomp_multi, linucb_one, linucb_multi


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{} ({}, {}) {2.2f} sec'.format(
              (method.__name__, args, kw, te - ts)))
        return result

    return timed


def timeit2(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed


def get_data(dataset):
    file_path = os.getcwd()
    d = os.path.join(os.sep, file_path, 'datasets/avazu')
    directory = os.path.join(os.sep, d, dataset)
    contexts = np.array(h5py.File(os.path.join(directory, 'X.hdf5'), 'r')['X'])
    reward_list = np.array(h5py.File(os.path.join(directory, 'Y.hdf5'), 'r')['Y'])
    users = h5py.File(os.path.join(directory, 'users.hdf5'), 'r')['users']
    users = np.array(users[:, 0])  # fix this
    return contexts, reward_list, users


def policy_generation(bandit, dim, t, numUsers):

    if bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(numUsers, d=dim, gamma=0.1, delta=0.1,
                                    R=0.02, epsilon=1/np.log(t/numUsers),
                                    random_state=3)

    elif bandit == 'ThompsonOne1':
        policy = thomp_one.ThompsonOne(d=dim, random_state=None, v=0.1)

    elif bandit == 'ThompMulti1':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, random_state=None, v=0.1)

    elif bandit == 'ThompMulti0':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, random_state=None, v=0)

    elif bandit == 'LinUcbOne1':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0.1)

    elif bandit == 'LinUcbMulti1':
        policy = linucb_multi.LinUcbMulti(numUsers, d=dim, alpha=0.4)

    elif bandit == 'random':
        policy = 0

    return policy


def policy_evaluation(policy, bandit, X, Y, users):

    print(bandit)
    T, k, d = X.shape
    print("k: {}".format(k))
    print("d: {}".format(d))

    seq_error = [0] * T

    if bandit in ['ThompCab', 'ThompMulti0', 'ThompMulti1',
                  'LinUcbMulti1']:
        for t in range(T):

            if t % 10000 == 0:
                print(t)

            user = users[t]
            full_context = X[t]
            action_id = policy.get_action(full_context, user)
            reward = Y[t, action_id]

            # update
            policy.reward(full_context[action_id], reward, action_id, user)
            if not reward:
                if t == 0:
                    seq_error[t] = 1
                else:
                    seq_error[t] = seq_error[t - 1] + 1
            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit in ['ThompsonOne1', 'LinUcbOne1']:
        for t in range(T):

            if t % 10000 == 0:
                print(t)

            user = users[t]
            full_context = X[t]
            action_id = policy.get_action(full_context)
            reward = Y[t, action_id]

            if not reward:
                policy.reward(full_context[action_id], reward, action_id)
                if t == 0:
                    seq_error[t] = 1
                else:
                    seq_error[t] = seq_error[t - 1] + 1
            else:
                policy.reward(full_context[action_id], reward, action_id)
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

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
    parser = argparse.ArgumentParser(
        description='Multi Armed Bandit algorithms on Avazu.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to use')
    args = parser.parse_args()

    # loading dataset
    dataset = args.dataset[0]
    dataset_path = os.path.join(
        os.sep, os.getcwd(), 'datasets/avazu/' + dataset)
    info_file = os.path.join(os.sep, dataset_path, 'info.txt')
    for i, line in enumerate(open(info_file, 'r')):
        if i == 0:
            k = int(line.split()[0])
        elif i == 1:
            numUsers = int(line.split()[2])
        # elif i == 3:
        #     time = int(line.split()[0]) / k
        # elif i == 4:
        #     d = int(line.split()[0])
        print(line.rstrip())

    X, Y, users = get_data(dataset)
    time, k, d = X.shape

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

    info = open(os.path.join(result_dir, 'info.txt'), 'w')
    # run algorithms
    bandits = ['ThompMulti0', 'ThompMulti1',
               'LinUcbMulti1', 'ThompsonOne1', 'LinUcbOne1']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, d, time, numUsers)
        seq_error = policy_evaluation(policy, bandit, X, Y, users)
        regret[bandit] = regret_calculation(seq_error)
        cum_regret[bandit] = seq_error
        # save results
        fileObject = open(os.path.join(cum_regret_dir, bandit + '.plot'), 'wb')
        regretObject = open(os.path.join(regret_dir, bandit + '.plot'), 'wb')
        if bandit is not 'random':
            info.write(policy.verbose())
        pickle.dump(cum_regret[bandit], fileObject)
        pickle.dump(regret[bandit], regretObject)
        fileObject.close()
        regretObject.close()


if __name__ == '__main__':
    main()
