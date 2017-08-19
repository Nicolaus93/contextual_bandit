import numpy as np
import argparse
from sklearn.preprocessing import normalize
import h5py
import os


def get_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state


def artificial_data_generator(T=1000, d=10, K=10, classes=5, numUsers=1,
                              model=None, random_state=None):

    r = get_random_state(random_state=random_state)

    # generate model
    if model is None:
        # model = np.random.randint(low=-1, high=2, size=(classes, d))
        model = np.random.random((classes, d))
        # model = np.random.randint(2, size=(classes, d))
        # model[model == 0] = -1
    else:
        classes, d = model.shape

    # generate users_feat
    users_feat = np.zeros(shape=(classes * numUsers, d))
    k = 0
    for i in range(classes):
        for j in range(numUsers):
            # add noise
            # a = np.random.rand(d) / 10
            # users_feat[k, :] = (model[i,:]+a) / np.sum(model[i,:]+a)
            users_feat[k, :] = model[i, :]
            k += 1

    # generate users
    users = np.random.randint(classes * numUsers, size=T)

    # generate users
    users = []
    tot = classes * numUsers
    for i in range(tot):
        users += [i] * T

    users = np.random.permutation(np.asarray(users))
    T = T * tot

    # generate contexts
    X = np.zeros(shape=(T, d, K))
    for t in range(T):
        # Xt = r.randint(10, size=(d, K))
        # Xt[Xt == 0] = -1
        # Xt = np.random.random((d, K))

        # sample uniformly between -1 and 1
        Xt = 2 * np.random.random_sample((d, K)) - 1

        # sample uniformly between 0 and 1
        # Xt = np.random.random_sample((d, K))

        # normalize
        X[t, :, :] = normalize(Xt, axis=0)

    # generate rewards
    # it could be done in the previous for loop
    # it's done here for clarity
    Y = np.zeros(shape=(T, K))
    for t in range(T):
        mean = users_feat[users[t]].dot(X[t, :, :])
        for i in range(K):
            Y[t, i] = r.normal(mean[i], 0.1)

    return X, Y, users, model, users_feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic data')
    parser.add_argument('-save', dest='save', action='store_true',
                        default=False,
                        help='whether to save data')
    parser.add_argument('-k', dest='k', metavar='items_per_round',
                        type=int, nargs=1,
                        help='number of items per round')
    parser.add_argument('-d', dest='d', metavar='features',
                        type=int, nargs=1,
                        help='dimension of features')
    parser.add_argument('-t', dest='t', metavar='interactions',
                        type=int, nargs=1,
                        help='interactions per user')
    parser.add_argument('-c', dest='c', metavar='classes',
                        type=int, nargs=1,
                        help='number of classes')
    parser.add_argument('-u', dest='u', metavar='users',
                        type=int, nargs=1,
                        help='number of users per class')

    args = parser.parse_args()
    save = args.save
    k = args.k[0]
    d = args.d[0]
    t = args.t[0]
    c = args.c[0]
    u = args.u[0]

    X, Y, users, model, us_feat = artificial_data_generator(T=t, d=d, K=k,
                                                            classes=c,
                                                            numUsers=u)
    if save:
        file_path = os.getcwd()
        dataset = str(c) + "_classes" + str(u) + "_users" + str(t) + "_rounds"
        directory = os.path.join(file_path, dataset)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("saving..")
        data = h5py.File(os.path.join(directory, 'dataset.hdf5'), 'w')
        data.create_dataset('X', data=X)
        data.create_dataset('y', data=Y)
        data.create_dataset('users', data=users)
        data.close()
