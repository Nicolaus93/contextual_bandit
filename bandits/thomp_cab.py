"""
Context aware clustering of bandits based on Thompson sampling
"""

import numpy as np
import random
from .utils import get_random_state, sherman_morrison


class ThompCAB(object):
    r"""
    Context aware clustering of bandits using Thompson Sampling

    Parameters
        ----------
        numUsers : int
            Number of users considered.

        d : int
            Dimensionality of each context.

        gamma : float
            Parameter used for updating the cluster.

        delta: float, 0 < delta < 1
            With probability 1 - delta, LinThompSamp satisfies the theoretical
            regret bound.

        R: float, R >= 0
            Assume that the residual  :math:`ri(t) - bi(t)^T \hat{\mu}`
            is R-sub-gaussian. In this case, R^2 represents the variance for
            residuals of the linear model :math:`bi(t)^T`.

        epsilon: float, 0 < epsilon < 1
            A  parameter  used  by  the  Thompson Sampling algorithm.
            If the total trials T is known, we can choose epsilon = 1/ln(T).

        random_state: {int, np.random.RandomState} (default: None)
            If int, np.random.RandomState will use it as seed. If None, a
            random seed will be used.

    References: to do

    """

    def __init__(self, users, d=128, gamma=0.5,
                 v=0.1, random_state=None):

        self.numUsers = users
        self.d = d
        self.random_state = get_random_state(random_state)
        self.gamma = gamma
        self.t = 0
        self.used = np.zeros(self.numUsers)
        self.updated = set()
        self.neigh_size = []
        # Thompson sampling parameters
        self.v = v
        self._init_model()

    def _init_model(self):
        self.mu_hat = np.zeros(shape=(self.numUsers, self.d))
        self.mu_tilde = np.zeros(shape=(self.numUsers, self.d))
        self.f = np.zeros(shape=(self.numUsers, self.d))
        self.B_inv = np.zeros(shape=(self.numUsers, self.d, self.d))
        for i, mat in enumerate(self.B_inv):
            self.B_inv[i] = np.eye(self.d)

    # @profile
    def _compute_neighbourood(self, context_array, user):
        """
        """
        K = len(context_array)
        # define neighborood sets
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)

        j = np.asarray(list(self.updated), dtype=np.int)
        seq = list(map(self.random_state.multivariate_normal,
                       self.mu_hat[j],
                       self.v**2 * self.B_inv[j]))
        self.mu_tilde[j] = np.asarray(seq)

        estimated_reward = self.mu_hat.dot(context_array.T)
        payoff = self.mu_tilde.dot(context_array.T)
        self.CB = np.abs(payoff - estimated_reward)
        i, j = np.where(np.abs(estimated_reward - estimated_reward[user]) <
                        self.CB[user] + self.CB)
        self.N[i, j] = 1
        # set users never used to 0
        t = np.where(self.used == 0)[0]
        self.N[t, :] = 0
        self.N[user, :] = 1

    # @profile
    def _compute_neigh_vectorized(self, context_array, user):
        """
        context_array: row vector
        user: user_id
        """
        K = len(context_array)
        # define neighborood sets
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)

        # self.neigh_size = 50
        # inds = np.where(self.used)[0]
        # to_sample = np.random.permutation(inds)[:self.neigh_size]
        # for j in to_sample:
        #     self.mu_tilde[j] = self.random_state.multivariate_normal(
        #                       self.mu_hat[j].flat, self.v**2 * self.B_inv[j])

        # sample mu tilde
        for j in self.updated:
            self.mu_tilde[j] = self.random_state.multivariate_normal(
                self.mu_hat[j].flat, self.v**2 * self.B_inv[j])

        """
        # # using indices for users
        # inds = np.where(self.used)[0]
        # u = np.where(inds==user)[0][0]
        # estimated_reward = context_array.dot(self.mu_hat[inds].T)
        # payoff = context_array.dot(self.mu_tilde[inds].T)
        # self.CB[:, inds] = np.abs(payoff - estimated_reward)
        # i, j = np.where( np.abs(estimated_reward -
                            estimated_reward[:,u][..., np.newaxis]) <
        #                 self.CB[:, u][..., np.newaxis] + self.CB[:, inds] )

        # self.N[inds[j], i] = 1
        # self.CB = self.CB.T
        """

        estimated_reward = self.mu_hat.dot(context_array.T)
        payoff = self.mu_tilde.dot(context_array.T)
        self.CB = np.abs(payoff - estimated_reward)
        i, j = np.where(np.abs(estimated_reward - estimated_reward[user]) <
                        self.CB[user] + self.CB)
        # i, j = np.where(np.abs(estimated_reward -
        #                 estimated_reward[user]) < 0.01)
        # self.CB = np.sqrt( np.square(payoff) - np.square(estimated_reward) )
        # i, j = np.where( np.sqrt(np.square(estimated_reward) -
        #                  estimated_reward[user]**2) <
        #     self.CB[user] + self.CB)
        self.N[i, j] = 1
        # set users never used to 0
        t = np.where(self.used == 0)[0]
        self.N[t, :] = 0
        self.N[user, :] = 1

    # @profile
    def _compute_neigh_vectorized2(self, context_array, user):
        """
        context_array: row vector
        user: user_id
        here consider payoff when calculating neighborood
        """
        K = len(context_array)
        # define neighborood sets
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)

        # sample mu tilde
        for j in self.updated:
            self.mu_tilde[j] = np.random.multivariate_normal(
                self.mu_hat[j].flat, self.v**2 * self.B_inv[j])

        estimated_reward = self.mu_hat.dot(context_array.T)
        payoff = self.mu_tilde.dot(context_array.T)
        self.CB = np.abs(payoff - estimated_reward)
        i, j = np.where(np.abs(payoff - payoff[user]) <
                        self.CB[user] + self.CB)
        self.N[i, j] = 1

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
        # update parameters in order to sample user's mu tilde
        self.used[user] += 1
        self.updated.add(user)
        self._compute_neigh_vectorized(context_array, user)
        avg_mu_tilde = self.mu_tilde.T.dot(self.N) / np.sum(self.N, axis=0)
        payoff = np.sum(np.multiply(context_array, avg_mu_tilde.T),
                        axis=1)  # is it right?
        action_id = np.argmax(payoff)
        return action_id

    def reward(self, x, reward, action_id):
        """
        Update the model
        """
        user = self.user
        self.updated = set()
        self._update(user, x, reward)
        self.neigh_size.append(sum(self.N[:, action_id]))
        if self.CB[user, action_id] <= self.gamma / 4:
            self.N[user, action_id] = 0
            to_update = self.CB[:, action_id] * self.N[:, action_id]
            for j, value in enumerate(to_update):
                if value <= self.gamma / 4 and value > 0:
                    self._update(j, x, reward)

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
        self.f[user] += reward * x
        B_inv = sherman_morrison(self.B_inv[user], x)
        self.B_inv[user] = B_inv
        self.mu_hat[user] = B_inv.dot(self.f[user])

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
        verbose = self.__class__.__name__ + ", v: " + str(self.v)
        return verbose
