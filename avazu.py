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
import argparse
import pickle
import h5py
from os.path import join, exists
from os import makedirs, getcwd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from bandits import thomp_cab, thomp_one, thomp_multi, linucb_one, linucb_multi, cab


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{}  {:2.2f} sec'.format(method.__name__, (te - ts)))
        return result
    return timed


def get_data(dataset):
    file_path = getcwd()
    d = join(file_path, 'datasets/avazu')
    directory = join(d, dataset)
    data = h5py.File(join(directory, 'dataset.hdf5'), 'r')
    contexts = np.array(data['X'])
    rewards = np.array(data['y'])
    users = np.array(data['users'])
    users = users[:, 0]
    return contexts, rewards, users


def policy_generation(bandit, dim, t, numUsers):

    if bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(numUsers, d=dim, gamma=0.1, v=0.01)

    elif bandit == 'ThompsonOne':
        policy = thomp_one.ThompsonOne(d=dim, random_state=None, v=0.1)

    elif bandit == 'ThompMulti':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, v=0.01)

    elif bandit == 'LinUcbOne':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0.1)

    elif bandit == 'LinUcbMulti':
        policy = linucb_multi.LinUcbMulti(numUsers, d=dim, alpha=0.0001)

    elif bandit == 'ExploitMulti':
        policy = linucb_multi.LinUcbMulti(numUsers, d=dim, alpha=0)

    elif bandit == 'ExploitSingle':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0)

    elif bandit == 'Cab':
        policy = cab.CAB(numUsers, d=dim, gamma=0.1, alpha=0.1)

    elif bandit == 'random':
        policy = 0

    return policy


@timeit
def policy_evaluation(policy, bandit, X, Y, users):

    print(bandit)
    T, k, d = X.shape
    seq_error = [0] * T

    # X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
    #     X, Y, users, test_size=0.8, random_state=42)

    # T, k, d = X_test.shape
    # seq_error = [0] * T

    for t in range(T):
        if bandit is not 'random':
            if t % 10000 == 0:
                print(t)

            policy.set_user(users[t])
            full_context = normalize(X[t])
            # policy.set_user(user_test[t])
            # full_context = normalize(X_test[t])
            action_id = policy.get_action(full_context)
            reward = Y[t, action_id]
            # reward = y_test[t, action_id]
            policy.reward(full_context[action_id], reward, action_id)

        else:
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
    parser.add_argument(dest='name', metavar='name', type=str, nargs=1,
                        help='name of the folder where results are saved')
    args = parser.parse_args()

    # loading dataset
    dataset = args.dataset[0]
    name = args.name[0]
    dataset_path = join(getcwd(), 'datasets/avazu/' + dataset)
    info_file = join(dataset_path, 'info.txt')
    for i, line in enumerate(open(info_file, 'r')):
        if i == 0:
            numUsers = int(line.split()[0])
        print(line.rstrip())
    print('\n')

    X, Y, users = get_data(dataset)
    X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
        X, Y, users, test_size=0.8, random_state=42)

    T, k, d = X_test.shape

    # create result directory
    result_dir = join(dataset_path, 'results')
    if not exists(result_dir):
        makedirs(result_dir)

    # create folder for the current test
    test_dir = join(result_dir, name)
    if not exists(test_dir):
        makedirs(join(result_dir, name))

    # create regret directory
    regret_dir = join(test_dir, 'regret')
    if not exists(regret_dir):
        makedirs(regret_dir)
    # create cum_regret directory
    cum_regret_dir = join(test_dir, 'cum_regret')
    if not exists(cum_regret_dir):
        makedirs(cum_regret_dir)

    # conduct regret analyses
    regret = {}
    cum_regret = {}

    info_path = join(test_dir, 'info.txt')
    info = open(info_path, 'w')

    # run algorithms
    bandits = ['random', 'ThompCab', 'ThompMulti', 'ThompsonOne',
               'ExploitMulti', 'LinUcbMulti', 'LinUcbOne', 'ExploitSingle',
               'Cab']
    for i, bandit in enumerate(bandits):

        # average regret in randomized algorithms
        if bandit in ['ThompCab', 'ThompMulti', 'ThompsonOne']:

            rounds = 5
            cum_regret[bandit] = np.zeros((rounds, T))
            regret[bandit] = np.zeros((rounds, T))
            for j in range(rounds):
                policy = policy_generation(bandit, d, T, numUsers)
                cum_regret[bandit][j] = policy_evaluation(
                    policy, bandit, X_test, y_test, user_test)
                regret[bandit][j] = regret_calculation(cum_regret[bandit][j])
                print("regret: {:.2f}\n".format(cum_regret[bandit][j, T - 1]))

        else:
            policy = policy_generation(bandit, d, T, numUsers)
            seq_error = policy_evaluation(
                policy, bandit, X_test, y_test, user_test)
            regret[bandit] = regret_calculation(seq_error)
            cum_regret[bandit] = seq_error
            print("regret: {:.2f}\n".format(cum_regret[bandit][T - 1]))

        # save results
        fileObject = open(join(cum_regret_dir, bandit + '.plot'), 'wb')
        regretObject = open(join(regret_dir, bandit + '.plot'), 'wb')
        if bandit is not 'random':
            info.write(policy.verbose())
            info.write('\n')
        pickle.dump(cum_regret[bandit], fileObject)
        pickle.dump(regret[bandit], regretObject)
        fileObject.close()
        regretObject.close()


if __name__ == '__main__':
    main()
