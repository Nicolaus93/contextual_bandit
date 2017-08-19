"""
Context aware clustering of bandits
"""

import numpy as np
from .utils import sherman_morrison


class CAB(object):
    r"""
    Context aware clustering of bandits

    Parameters:


    References:
    ----------
    .. [1]

    """

    def __init__(self, users, d=128, gamma=0.5, alpha=0.5, minUsed=0, p=0.2):

        self.numUsers = users  # number of Users
        self.d = d
        self.alpha = alpha
        self.minUsed = minUsed
        self.gamma = gamma
        self.p = p
        self.t = 0
        self.used = np.zeros((self.numUsers), dtype=int)  # for user inverse
        self._init_model()
        # for plots
        self.neigh_size = []
        self.updated_size = []

    def _init_model(self):
        """
        initiate model for each user
        """
        self.A_inv = np.zeros(shape=(self.numUsers, self.d, self.d))
        self.b = np.zeros(shape=(self.numUsers, self.d))
        self.w = np.zeros(shape=(self.numUsers, self.d))
        for i, mat in enumerate(self.A_inv):
            self.A_inv[i] = np.eye(self.d)

    def _neighbourhood(self, context_array, user):
        """
        context_array: row vector
        user: user_id
        """
        K = len(context_array)
        # define confidence bounds
        self.CB = np.zeros(shape=(self.numUsers, K))
        # define neighborood sets
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)
        # compute confidence bounds
        for u in range(self.numUsers):
            temp = self.alpha * np.sqrt(context_array.dot(self.A_inv[u])
                                        .dot(context_array.T))
            self.CB[u] = np.diag(temp)
        estimated_reward = self.w.dot(context_array.T)
        # find users in the neighborhood
        i, j = np.where(np.abs(estimated_reward - estimated_reward[user]) <
                        self.CB[user] + self.CB)
        self.N[i, j] = 1
        # set users never used to 0
        t = np.where(self.used == 0)[0]
        self.N[t, :] = 0
        # sanity check
        self.N[user, :] = 1

    def get_action(self, context_array):
        """Return the action to perform

        Parameters
        ----------
        context_array : numpy array
            K x d array containing contexts for every action.

        user : int
            Id of the user at the current round.

        Returns
        -------
        action_id : int
            Id of the action that will be recommended to the user.
            Currently it returns the FIRST best action!
        """
        user = self.user
        self.t += 1
        # update parameters
        self.used[user] += 1
        self._neighbourhood(context_array, user)
        avg_w = self.w.T.dot(self.N) / np.sum(self.N, axis=0)
        payoff = np.sum(np.multiply(context_array, avg_w.T), axis=1)
        action_id = np.argmax(payoff)
        return action_id

    def reward(self, x, reward, action_id):
        """
        Update the model
        """
        user = self.user
        self.updated = set()
        self._update(user, x, reward)  # update the user model first
        # update neighborhood size
        self.neigh_size.append(sum(self.N[:, action_id]))

        # update users in the neighborhood
        if self.CB[user, action_id] <= self.gamma:
            self.N[user, action_id] = 0  # user already updated
            to_update = self.CB[:, action_id] * self.N[:, action_id]
            for j, value in enumerate(to_update):
                if value <= self.gamma / self.used[j] \
                        and value > 0:
                    self._update(j, x, reward)

        # update updated size
        self.updated_size.append(len(self.updated))

    def _update(self, user, x, reward):
        """
        Update the model.

        Parameters
        ----------
        user : int
            id of the user

        x : np.array
            context of the action performed

        reward : float or int
            reward for the action taken
        """
        self.updated.add(user)
        self.b[user] += reward * x
        A_inv = sherman_morrison(self.A_inv[user], x)
        self.A_inv[user] = A_inv
        self.w[user] = A_inv.dot(self.b[user])

    def set_user(self, user):
        """
        Parameters
        ----------
        user : int
            id of the current user
        """
        self.user = user

    def verbose(self):
        """
        Return bandit name and parameters
        """
        verbose = self.__class__.__name__ + ", alpha: " + str(self.alpha)
        return verbose
