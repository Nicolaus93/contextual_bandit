# -*- coding: utf-8 -*-
"""
==============================
Contextual bandit on Avazu
==============================
The script uses real-world data to conduct contextual bandit experiments. Here we use
Avazu dataset, which is released by ... . Please fist pre-process
datasets (use ...), and then you can run this example.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from striatum.storage import history
from striatum.storage import model
from striatum.storage import action
from striatum.bandit import linthompsamp, linucb, cab, thomp_cab
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def get_data():
    streaming_batch = pd.read_csv('datasets/avazu/processed10k.csv')
    users = pd.read_csv('datasets/avazu/users.csv')
    reward_list = pd.read_csv('datasets/avazu/reward_list.csv')
    return streaming_batch, users, reward_list


def policy_generation(bandit):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    # actionstorage = action.MemoryActionStorage()
    actionstorage = list(range(10))

    if bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(historystorage, modelstorage, actionstorage,
                                           context_dimension=37, delta=0.1, R=0.01, epsilon=1/np.log(1000))

    elif bandit == 'LinUCB':
        policy = linucb.LinUCB(historystorage, modelstorage, actionstorage, alpha=0.3, context_dimension=37)

    elif bandit == 'Cab':
        policy = cab.CAB(historystorage, modelstorage, actionstorage, 6040, context_dimension=37, minUsed=0)

    elif bandit == 'random':
        policy = 0

    elif bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(historystorage, modelstorage, actionstorage, 6040, context_dimension=37, minUsed=1, 
                                        delta=0.1, R=0.01, epsilon=1/np.log(1000))

    return policy


def policy_evaluation(policy, bandit, streaming_batch, users, reward_list):
    k = 10
    times = len(streaming_batch) // k
    seq_error = np.zeros(shape=(times, 1))
    action_ids = range(k)

    if bandit in ['LinUCB', 'LinThompSamp', 'UCB1', 'Exp3']:
        print(bandit)
        for t in range(times):
            user = users.iloc[t*k][0]
            print(user) # debugging
            full_context = {}
            for action_id in action_ids:
                full_context[action_id] = np.array(streaming_batch.iloc[t*k+action_id])

            # get next (one) action to perform and its reward
            history_id, action = policy.get_action(full_context, 1)
            reward = reward_list.iloc[t*k+action[0]]
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
        print(bandit)
        for t in range(times):
            user = users.iloc[t*10]['device_ip']
            # print('user: ' + str(user)) # debugging
            full_context = {}
            for action_id in action_ids:
                full_context[action_id] = np.array(streaming_batch.iloc[t*k+action_id][1:])

            history_id, action = policy.get_action(full_context, user)
            # print('action: ' + str(action[0].action)) # debugging again
            reward = reward_list.iloc[t*10+action[0].action]['click']
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
            action = actions_id[np.random.randint(0, len(actions)-1)]
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]
            if action not in list(watched_list['movie_id']):
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
    streaming_batch, users, reward_list = get_data()
    # conduct regret analyses
    regret = {}
    col = ['b', 'g', 'r', 'y']
    # bandits = ['LinUCB']
    # bandits = ['Cab']
    # bandits = ['ThompCab', 'Cab', 'LinThompSamp', 'LinUCB']
    bandits = ['ThompCab']
    # bandits = ['LinThompSamp', 'random']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit)
        seq_error = policy_evaluation(policy, bandit, streaming_batch, users, reward_list)
        regret[bandit] = regret_calculation(seq_error)
        plt.plot(range(len(streaming_batch)//10), regret[bandit], c=col[i], ls='-', label=bandit)
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T")
    plt.show()


if __name__ == '__main__':
    main()