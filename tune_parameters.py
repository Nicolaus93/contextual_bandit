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


def policy_generation(bandit, dim, t, numUsers, param):

    if bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(
            numUsers, d=dim, gamma=0.1, v=param)

    elif bandit == 'ThompMulti':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, v=param)

    elif bandit == 'ThompsonOne':
        policy = thomp_one.ThompsonOne(d=dim, v=param)

    elif bandit == 'LinUcbOne':
        policy = linucb_one.LinUcbOne(d=dim, alpha=param)

    elif bandit == 'LinUcbMulti':
        policy = linucb_multi.LinUcbMulti(numUsers, d=dim, alpha=param)

    elif bandit == 'Cab':
        policy = cab.CAB(numUsers, d=dim, gamma=0.1, alpha=param)

    return policy


def explore_parameters(bandit, dim, t, numUsers):
    """
    A function used to tune various parameters
    in the algorithms.
    """

    test = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    policy = {}

    if bandit == 'ThompCab':
        for i in test:
            policy[bandit + str(i)] = thomp_cab.ThompCAB(
                numUsers, d=dim, gamma=0.1, v=i)

    elif bandit == 'ThompsonOne':
        for i in test:
            policy[bandit + str(i)] = thomp_one.ThompsonOne(d=dim, v=i)

    elif bandit == 'ThompMulti':
        for i in test:
            policy[bandit + str(i)] = thomp_multi.ThompMulti(
                numUsers, d=dim, v=i)

    elif bandit == 'LinUcbOne':
        for i in test:
            policy[bandit + str(i)] = linucb_one.LinUcbOne(d=dim, alpha=i)

    elif bandit == 'LinUcbMulti':
        for i in test:
            policy[bandit + str(i)] = linucb_multi.LinUcbMulti(
                numUsers, d=dim, alpha=i)

    elif bandit == 'Cab':
        for i in test:
            policy[bandit + str(i)] = cab.CAB(
                numUsers, d=dim, gamma=0.1, alpha=i)

    return policy


@timeit
def policy_evaluation(policy, bandit, X, y, users):

    policy.verbose()

    T, k, d = X.shape
    seq_error = [0] * T

    for t in range(T):
        if bandit is not 'random':
            if t % 10000 == 0:
                print(t)

            policy.set_user(users[t])
            full_context = normalize(X[t])
            action_id = policy.get_action(full_context)
            reward = y[t, action_id]
            policy.reward(full_context[action_id], reward, action_id)

        else:
            action_id = np.random.randint(0, k)
            reward = y[t, action_id]

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

    X, Y, users = get_data(dataset)
    X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
        X, Y, users, test_size=0.8, random_state=42)

    T, k, d = X_train.shape

    # create result directory
    result_dir = join(dataset_path, 'results')
    if not exists(result_dir):
        makedirs(result_dir)

    # create folder for the current test
    test_dir = join(result_dir, name)
    if not exists(test_dir):
        makedirs(join(result_dir, name))

    # conduct regret analyses
    regret = {}
    cum_regret = {}

    # conduct regret analyses
    regret = {}
    cum_regret = {}
    bandits = ['ThompMulti', 'ThompsonOne',
               'LinUcbMulti', 'LinUcbOne', 'ThompCab']
    for b in bandits:

        res_path = join(test_dir, b)
        makedirs(res_path)
        # create regret directory
        regret_dir = join(res_path, 'regret')
        makedirs(regret_dir)
        # create cum_regret directory
        cum_regret_dir = join(res_path, 'cum_regret')
        makedirs(cum_regret_dir)
        # create info file
        info_path = join(res_path, 'info.txt')
        info = open(info_path, 'w')

        # policies = explore_parameters(b, d, T, numUsers)

        params = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        for i in params:

            name = b + str(i)

            if b in ['ThompCab', 'ThompMulti', 'ThompsonOne']:

                rounds = 10
                cum_regret[name] = np.zeros((rounds, T))
                regret[name] = np.zeros((rounds, T))
                for j in range(rounds):
                    policy = policy_generation(b, d, T, numUsers, i)
                    try:
                        cum_regret[name][j] = policy_evaluation(
                            policy, b, X_train, y_train, user_train)
                    except Exception as e:
                        print(e)
                    regret[name][j] = regret_calculation(cum_regret[name][j])

            else:
                policy = policy_generation(b, d, T, numUsers, i)
                cum_regret[name] = policy_evaluation(
                    policy, b, X_train, y_train, user_train)
                regret[name] = regret_calculation(cum_regret[name])

            # save results
            fileObject = open(join(cum_regret_dir, name + '.plot'), 'wb')
            regretObject = open(join(regret_dir, name + '.plot'), 'wb')
            info.write(policy.verbose())
            info.write('\n')
            pickle.dump(cum_regret[name], fileObject)
            pickle.dump(regret[name], regretObject)
            fileObject.close()
            regretObject.close()

        # # run tests
        # for bandit in policies:

        #     name = bandit.split('0')[0]
        #     # average regret in randomized algorithms
        #     if name in ['ThompCab', 'ThompMulti', 'ThompsonOne']:
        #         rounds = 10
        #         cum_regret[bandit] = np.zeros((rounds, T))
        #         regret[bandit] = np.zeros((rounds, T))
        #         for j in range(rounds):
        #             policy = policy_generation(bandit, d, T, numUsers)
        #             cum_regret[bandit][j] = policy_evaluation(
        #                 policy, bandit, X_train, y_train, user_train)
        #             regret[bandit][j] = regret_calculation(cum_regret[bandit][j])
        #             print("regret: {:.2f}\n".format(cum_regret[bandit][j, T - 1]))

        #     seq_error = policy_evaluation(policies[bandit], bandit, X_train, y_train, user_train)
        #     regret[bandit] = regret_calculation(seq_error)
        #     cum_regret[bandit] = seq_error
        #     # save results
        #     fileObject = open(join(cum_regret_dir, bandit + '.plot'), 'wb')
        #     regretObject = open(join(regret_dir, bandit + '.plot'), 'wb')
        #     if bandit is not 'random':
        #         info.write(policies[bandit].verbose())
        #         info.write('\n')
        #     pickle.dump(cum_regret[bandit], fileObject)
        #     pickle.dump(regret[bandit], regretObject)
        #     fileObject.close()
        #     regretObject.close()


if __name__ == '__main__':
    main()
