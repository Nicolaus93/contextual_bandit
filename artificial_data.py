import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import time
from datasets.artificial_data.generate_data import artificial_data_generator
from bandits import thomp_cab, thomp_multi, thomp_one, linucb_one, linucb_multi
from sklearn.preprocessing import normalize


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


def policy_generation(bandit, dim, t, numUsers):

    if bandit == 'ExploitOne':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0)

    if bandit == 'LinUcbOne':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0.2)

    if bandit == 'ThompCAB':
        policy = thomp_cab.ThompCAB(numUsers, d=dim, gamma=0.1,
                                    v=0.1)

    elif bandit == 'ThompMulti':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, random_state=None, v=0.1)

    elif bandit == 'ExploitMulti':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, random_state=None, v=0)

    elif bandit == 'ThompsonOne':
        policy = thomp_one.ThompsonOne(d=dim, random_state=None, v=0.1)

    elif bandit == 'LinUcbMulti':
        policy = linucb_multi.LinUcbMulti(numUsers, d=dim, alpha=0.8)

    elif bandit == 'random':
        policy = 0

    return policy


@timeit
def policy_evaluation(policy, bandit, X, Y, users):

    print(bandit)
    T, d, k = X.shape
    seq_error = [0] * T

    for t in range(T):
        if bandit is not 'random':
            if t % 10000 == 0:
                print(t)

            policy.set_user(users[t])
            full_context = normalize(X[t].T)
            action_id = policy.get_action(full_context)
            reward = Y[t, action_id]
            policy.reward(full_context[action_id], reward, action_id)

        else:
            action_id = np.random.randint(0, k)
            reward = Y[t, action_id]

        if t == 0:
            seq_error[t] = max(Y[t]) - reward
        else:
            seq_error[t] = seq_error[t - 1] + max(Y[t]) - reward

    return seq_error


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    # model = np.array([[5, 1, 4, 1, 1, 1, 4, 1, 2, 4],
    #                   [5, 1, 1, 4, 2, 1, 5, 1, 5, 1],
    #                   [1, 5, 1, 5, 1, 4, 1, 4, 2, 1],
    #                   [1, 5, 4, 2, 5, 1, 1, 5, 1, 1],
    #                   [2, 1, 1, 1, 5, 4, 1, 1, 4, 5]])

    # model = 1 * (model - 2)

    # X, Y, users, model, users_feat = artificial_data_generator(
    #     T=10000, K=20, numUsers=40, model=model, random_state=42)

    classes = 5
    us = 200
    X, Y, users, model, users_feat = artificial_data_generator(
        T=10000, d=30, K=20, classes=classes, numUsers=us)

    T, d, k = X.shape
    classes = len(model)
    numUsers = len(np.unique(users))
    print('number of users: {}'.format(numUsers))

    cum_regret = {}
    regret = {}
    col = ['b', 'g', 'r', 'y', 'm', 'k']

    # run algorithms
    bandits = ['ThompMulti', 'random', 'ThompCAB',
               'ThompsonOne', 'LinUcbMulti', 'ExploitMulti']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, d, T, numUsers)
        seq_error = policy_evaluation(policy, bandit, X, Y, users)
        cum_regret[bandit] = seq_error
        regret[bandit] = regret_calculation(seq_error)
        t = len(cum_regret[bandit])

        plt.figure(1, figsize=(12, 8))
        plt.plot(range(t), cum_regret[bandit], c=col[i],
                 ls='-', label=str(bandit), linewidth=1.5)
        p1 = plt.legend(loc='upper left')
        plt.xlabel('time')
        plt.ylabel('cumulative regret')

        plt.figure(2, figsize=(10, 7))
        plt.plot(range(t), regret[bandit], c=col[i],
                 ls='-', label=str(bandit), linewidth=1.3)
        plt.ylim([0, 1.5])
        p2 = plt.legend(loc='upper right')
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.title("Regret plot with respect to T")

        if bandit == 'ThompCAB':
            plt.figure(3)
            plt.plot(range(t), policy.neigh_size)
            plt.title('Neighborood size')

            plt.figure(4)
            plt.plot(range(t), policy.updated_size)
            plt.title('Updated users with respect to T')

    p1.draggable(True)
    p2.draggable(True)
    plt.show()


if __name__ == '__main__':
    main()
