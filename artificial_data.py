import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import time
from bandits import thomp_cab, thomp_one, thomp_multi, linucb_one, linucb_multi, cab
from sklearn.preprocessing import normalize
import argparse
import os
import h5py
import scipy.stats as st


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
        policy = thomp_cab.ThompCAB(numUsers, d=dim, gamma=0.1, v=0.1)

    elif bandit == 'ThompMulti':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, random_state=None, v=0.1)

    elif bandit == 'ExploitMulti':
        policy = thomp_multi.ThompMulti(
            numUsers, d=dim, random_state=None, v=0)

    elif bandit == 'ThompsonOne':
        policy = thomp_one.ThompsonOne(d=dim, random_state=None, v=0.1)

    elif bandit == 'LinUcbMulti':
        policy = linucb_multi.LinUcbMulti(numUsers, d=dim, alpha=0.1)

    elif bandit == 'CAB':
        policy = cab.CAB(numUsers, d=dim, gamma=0.1, alpha=0.1)

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


def get_data(dataset):
    file_path = os.getcwd()
    d = os.path.join(file_path, 'datasets/avazu')
    directory = os.path.join(d, dataset)
    data = h5py.File(os.path.join(directory, 'dataset.hdf5'), 'r')
    contexts = np.array(data['X'])
    rewards = np.array(data['y'])
    users = np.array(data['users'])
    model = np.array(data['model'])
    return contexts, rewards, users, model


def plot_regret(fig, x, y, name, col, ylabel):
    plt.figure(fig, figsize=(12, 8))
    plt.plot(x, y, c=col,
             ls='-', label=name, linewidth=1.5)
    plt.xlabel('time')
    plt.ylabel(ylabel)
    return plt


def main():
    parser = argparse.ArgumentParser(
        description='Multi Armed Bandit algorithms on Avazu.')
    parser.add_argument(dest='dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset to use')
    args = parser.parse_args()

    # loading dataset
    dataset = args.dataset[0]
    dataset_path = os.path.join(
        os.getcwd(), 'datasets/artificial_data/data/' + dataset)
    print(dataset_path)

    # X, Y, users, model, us_feat = get_data(dataset_path)
    X, Y, users, model = get_data(dataset_path)
    T, d, k = X.shape
    classes = len(model)
    numUsers = len(np.unique(users))
    print('number of classes: {}'.format(classes))
    print('number of users per class: {}'.format(numUsers // classes))
    print('rounds per user: {}'.format(T // numUsers))
    print('arms per round: {}'.format(k))
    print('dimension of contexts: {}'.format(d))
    print('\n')

    cum_regret = {}
    regret = {}
    col = seaborn.color_palette()

    # plot model
    if len(model) < 6:
        plt.figure(3)
        plt.title('model classes')
        i, j = model.shape
        first = int(str(i) + '11')  # plot x11
        ax1 = plt.subplot(first)
        plt.setp(ax1.get_xticklabels(), visible=False)
        x = range(d)
        plt.bar(x, height=model[0])
        for k in range(2, classes + 1):
            num = int(str(i) + '1' + str(k))
            plt.subplot(num, sharex=ax1)
            plt.bar(x, height=model[k - 1], color=col[k - 1])
            ax2 = plt.subplot(num, sharex=ax1)
            plt.setp(ax2.get_xticklabels(), visible=False)

    # run algorithms
    bandits = ['CAB', 'ThompCAB', 'ThompMulti', 'ThompsonOne']
    for i, bandit in enumerate(bandits):

        # average regret in randomized algorithms
        if bandit in ['ThompCAB', 'ThompMulti', 'ThompsonOne']:

            rounds = 5
            cum_regret = np.zeros((rounds, T))
            regret = np.zeros((rounds, T))
            for j in range(rounds):
                policy = policy_generation(bandit, d, T, numUsers)
                cum_regret[j] = policy_evaluation(policy, bandit, X, Y, users)
                regret[j] = regret_calculation(cum_regret[j])
                print("regret: {:.2f}\n".format(cum_regret[j, T - 1]))
            avg_cum_regret = np.mean(cum_regret, axis=0)
            avg_regret = np.mean(regret, axis=0)

            # plot regret
            plot_regret(1, range(T), avg_cum_regret, str(bandit),
                        col[i], 'cumulative regret')
            plot_regret(2, range(T), avg_regret, str(bandit),
                        col[i], 'regret')

            # plot confidence bounds
            # check here https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            # ii, jj = st.t.interval(
            #     0.99, len(avg_regret) - 1,
            #     loc=np.mean(avg_regret), scale=st.sem(regret))
            ii, jj = st.norm.interval(
                0.95, loc=avg_regret, scale=st.sem(regret))

            plt.fill(np.concatenate([range(T), range(T)[::-1]]),
                     np.concatenate([ii, (jj)[::-1]]),
                     alpha=.5, fc=col[i], ec='None',
                     label='95% confidence interval')
        else:
            policy = policy_generation(bandit, d, T, numUsers)
            cum_regret = policy_evaluation(policy, bandit, X, Y, users)
            regret = regret_calculation(cum_regret)
            print("regret: {:.2f}\n".format(cum_regret[T - 1]))

            plot_regret(1, range(T), cum_regret, str(bandit),
                        col[i], 'cumulative regret')
            plot_regret(2, range(T), regret, str(bandit),
                        col[i], 'regret')

        # plot neighborhood and updated size in clustering algorithms
        if bandit in ['CAB', 'ThompCAB']:

            if bandit == 'CAB':
                c = col[0]
            else:
                c = col[2]
                continue

            # Two subplots, the axes array is 1-d
            seaborn.set_style("darkgrid", {'axes.grid': True})
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot(range(T), policy.neigh_size, c=c)
            axarr[0].set_title(bandit + ': neighborhood size')
            axarr[1].plot(range(T), policy.updated_size, c=c)
            axarr[1].set_title(bandit + ': updated size')

            # add a subplot for common X and Y names
            # check here https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots#6981055
            seaborn.set_style("darkgrid", {'axes.grid': False})
            # add a big axes, hide frame
            f.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top='off',
                            bottom='off', left='off', right='off')
            plt.xlabel("Time")
            plt.ylabel("size")

    plt.figure(1)
    plt.legend(loc='upper left')
    plt.figure(2)
    plt.legend(loc='upper right')
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    main()
