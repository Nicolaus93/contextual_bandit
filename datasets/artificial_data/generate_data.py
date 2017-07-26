import numpy as np
from sklearn.preprocessing import normalize


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
        # model = np.random.randint(low=-1, high=2, size=(classes,d))
        model = np.random.randint(2, size=(classes, d))
        model[model == 0] = -1
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

    # generate contexts
    X = np.zeros(shape=(T, d, K))
    for t in range(T):
        Xt = r.randint(10, size=(d, K))
        Xt[Xt == 0] = -1
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
    X, Y, users, model, us_feat = artificial_data_generator(T=2, d=5, K=3,
                                                            classes=2,
                                                            numUsers=1)
