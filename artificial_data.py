import numpy as np
from datasets.artificial_data.generate_data import artificial_data_generator
from bandits import thomp_cab, thomp_multi, thomp_one, linucb_one
import matplotlib.pyplot as plt
import time


def policy_generation(bandit, dim, t, numUsers):

    if bandit == 'LinUcbOne0':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0)

    if bandit == 'LinUcbOne1':
        policy = linucb_one.LinUcbOne(d=dim, alpha=0.6)

    if bandit == 'ThompCAB':
        policy = thomp_cab.ThompCAB(numUsers, d=dim, gamma=0.05, delta=0.1,
                                    R=0.02, epsilon=1 / np.log(t / numUsers),
                                    random_state=4)

    elif bandit == 'ThompMulti':
        policy = thomp_multi.ThompMulti(numUsers, d=dim, delta=0.1, R=0.02,
                                        epsilon=1 / np.log(t / numUsers),
                                        random_state=4)

    elif bandit == 'ThompsonOne':
        policy = thomp_one.ThompsonOne(d=dim, delta=0.1, R=0.02,
                                       epsilon=1 / np.log(t),
                                       random_state=None)

    elif bandit == 'random':
        policy = 0

    return policy


def policy_evaluation(policy, bandit, X, Y, users):
    print(bandit)

    T, d, k = X.shape
    seq_error = [0] * T

    # timing
    start = time.time()

    if bandit in ['ThompCAB', 'ThompMulti']:
        for t in range(T):
            # if t % 100 == 0:
            #     print('round ' + str(t))

            user = users[t]
            full_context = X[t].T
            action_id = policy.get_action(full_context, user)
            reward = Y[t, action_id]

            # update policy
            policy.reward(full_context[action_id], reward, action_id, user)
            seq_error[t] = seq_error[t - 1] + max(Y[t]) - reward

    elif bandit in ['ThompsonOne', 'LinUcbOne1', 'LinUcbOne0']:
        for t in range(T):
            full_context = X[t].T
            action_id = policy.get_action(full_context)
            reward = Y[t, action_id]

            # update policy
            policy.reward(full_context[action_id], reward, action_id)
            seq_error[t] = seq_error[t - 1] + max(Y[t]) - reward

    elif bandit == 'random':
        for t in range(T):
            action_id = np.random.randint(0, k)
            reward = Y[t, action_id]
            seq_error[t] = seq_error[t - 1] + max(Y[t]) - reward

    end = time.time()
    print('time: {} sec'.format(end - start))

    return seq_error


def main():
    # model = 0.1 * np.array([[5,1,4,1,1,1,4,1,2,4],
    #                         [5,1,1,4,2,1,5,1,5,1],
    #                         [1,5,1,5,1,4,1,4,2,1],
    #                         [1,5,4,2,5,1,1,5,1,1],
    #                         [2,1,1,1,5,4,1,1,4,5]])

    # X, Y, users, model, users_feat = artificial_data_generator(T=10000, d=5, K=20, classes=4, numUsers=10, model=model)
    X, Y, users, model, users_feat = artificial_data_generator(T=1000,
                                                               d=30,
                                                               K=20,
                                                               classes=1,
                                                               numUsers=1)

    # print(model)
    # print("users feat")
    # print(users_feat)
    # print(Y)

    T, d, k = X.shape
    numUsers = len(np.unique(users))

    cum_regret = {}
    col = ['b', 'g', 'r', 'y', 'm']

    # run algorithms
    bandits = ['LinUcbOne0', 'LinUcbOne1', 'ThompsonOne', 'random']
    for i, bandit in enumerate(bandits):
        policy = policy_generation(bandit, d, T, numUsers)
        seq_error = policy_evaluation(policy, bandit, X, Y, users)
        cum_regret[bandit] = seq_error
        time = len(cum_regret[bandit])

        # if bandit == 'ThompCAB':
        #     plt.plot(range(time), policy.neigh_size)

        plt.plot(range(time),
                 cum_regret[bandit],
                 c=col[i],
                 ls='-',
                 label=str(bandit))
        plt.xlabel('time')
        plt.ylabel('regret')
        # axes = plt.gca()
        plt.title("Regret Bound with respect to T")

    p = plt.legend(loc='upper left')
    p.draggable(True)
    plt.show()


if __name__ == '__main__':
    main()
