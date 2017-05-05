# -*- coding: utf-8 -*-
"""
==============================
Contextual bandit on MovieLens
==============================
The script uses real-world data to conduct contextual bandit experiments. Here we use
MovieLens 10M Dataset, which is released by GroupLens at 1/2009. Please fist pre-process
datasets (use "movielens_preprocess.py"), and then you can run this example.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linthompsamp, linucb, cab, thomp_cab
from striatum.storage.action import Action
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def get_data():
    streaming_batch = pd.read_csv('datasets/movielens/streaming_batch.csv', sep='\t', names=['user_id'], engine='c')
    user_feature = pd.read_csv('datasets/movielens/user_feature.csv', sep='\t', header=0, index_col=0, engine='c')
    actions_id = list(pd.read_csv('datasets/movielens/actions.csv', sep='\t', header=0, engine='c')['movie_id'])
    reward_list = pd.read_csv('datasets/movielens/reward_list.csv', sep='\t', header=0, engine='c')
    action_context = pd.read_csv('datasets/movielens/action_context.csv', sep='\t', header=0, engine='c')

    actions = []
    for key in actions_id:
        # action = Action(key)
        # actions.append(action)
        actions.append(key)
    return streaming_batch, user_feature, actions, reward_list, action_context


def policy_generation(bandit, actions):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()

    if bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(historystorage, modelstorage, actions,
                                           context_dimension=18, delta=0.1, R=0.01, epsilon=1/np.log(1000))

    elif bandit == 'LinUCB':
        policy = linucb.LinUCB(historystorage, modelstorage, actions, alpha=0.3, context_dimension=18)

    elif bandit == 'Cab':
        policy = cab.CAB(historystorage, modelstorage, actions, 6040, context_dimension=18, minUsed=0)

    elif bandit == 'random':
        policy = 0

    elif bandit == 'ThompCab':
        policy = thomp_cab.ThompCAB(historystorage, modelstorage, actions, 6040, context_dimension=18, minUsed=0, 
                                        delta=0.1, R=0.01, epsilon=1/np.log(1000))

    return policy


def policy_evaluation(policy, bandit, streaming_batch, user_feature, reward_list, actions, action_context=None):
    times = len(streaming_batch)
    seq_error = np.zeros(shape=(times, 1))
    # actions_id = [actions[i].id for i in range(len(actions))]
    actions_id = [actions[i] for i in range(len(actions))]
    action_features = action_context.drop('movie_name', 1)

    if bandit in ['LinUCB', 'LinThompSamp', 'UCB1', 'Exp3']:
        print(bandit)
        for t in range(times):
            # features associated to the user: tags they've watched for non-top-50 movies normalized per user
            feature = np.array(user_feature[user_feature.index == streaming_batch.iloc[t, 0]])[0]

            print(streaming_batch.iloc[t, 0]) # user_id
            full_context = {} # for all actions: num_actions x 18
            old_context = {}
            for action_id in actions_id:
                action_feature = np.asarray(action_features[(action_features['movie_id']==action_id)].iloc[:,1:])
                full_context[action_id] = np.multiply(feature, action_feature)
                old_context = feature

            # get next (one) action to perform
            history_id, action = policy.get_action(full_context, 1)
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]
            # action is a list of recommendation (from recommendation class)
            # if action[0]['action'].action_id not in list(watched_list['movie_id']):
            if action[0].action not in list(watched_list['movie_id']):
                # policy.reward(history_id, {action[0]['action'].action_id: 0.0})
                policy.reward(history_id, {action[0].action: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                # policy.reward(history_id, {action[0]['action'].action_id: 1.0})
                policy.reward(history_id, {action[0].action: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit in ['Cab', 'ThompCab']:
        print(bandit)
        for t in range(times):
            # features associated to the user: tags they've watched for non-top-50 movies normalized per user
            feature = np.array(user_feature[user_feature.index == streaming_batch.iloc[t, 0]])[0]

            print(streaming_batch.iloc[t, 0])
            full_context = {} # for all actions: num_actions x 18
            for action_id in actions_id:
                action_feature = np.asarray(action_features[(action_features['movie_id']==action_id)].iloc[:,1:])
                full_context[action_id] = np.multiply(feature, action_feature)
                # full_context[action_id] = np.concatenate([feature, action_feature.flatten()])

            history_id, action = policy.get_action(full_context, streaming_batch.iloc[t, 0]-1, n_actions=1)
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]
            # action is a list of recommendation (from recommendation class)
            # if action[0]['action'].action_id not in list(watched_list['movie_id']):
            if action[0].action not in list(watched_list['movie_id']):
                # policy.reward(history_id, {action[0]['action'].action_id: 0.0})
                policy.reward(history_id, {action[0].action: 0.0}, streaming_batch.iloc[t, 0]-1)
                # policy.reward(history_id, action[0])
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                # policy.reward(history_id, {action[0]['action'].action_id: 1.0})
                policy.reward(history_id, {action[0].action: 1.0}, streaming_batch.iloc[t, 0]-1)
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

        # policy.get_parameters()

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
    """
    streaming_batch: sequence of users to serve
    user_feature: features for every user
    actions: ids of the 50 actions (movies) which can be recommended
    reward_list: rewards
    action_context: tags of the 50 movies which can be recommended
    """
    streaming_batch, user_feature, actions, reward_list, action_context = get_data()
    streaming_batch_small = streaming_batch.iloc[0:100]

    # conduct regret analyses
    regret = {}
    cum_regret = {}
    col = ['b', 'g', 'r', 'y']
    # bandits = ['LinUCB']
    # bandits = ['Cab']
    bandits = ['ThompCab', 'Cab', 'LinThompSamp', 'LinUCB']
    # bandits = ['LinThompSamp', 'random']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, actions)
        seq_error = policy_evaluation(policy, bandit, streaming_batch_small, user_feature, reward_list,
                                          actions, action_context)
        regret[bandit] = regret_calculation(seq_error)
        cum_regret[bandit] = seq_error
        # plt.plot(range(len(streaming_batch_small)), regret[bandit], c=col[i], ls='-', label=bandit)
        plt.plot(range(len(streaming_batch_small)), cum_regret[bandit], c=col[i], ls='-', label=bandit)
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        axes = plt.gca()
        # axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T")
    plt.show()


if __name__ == '__main__':
    main()