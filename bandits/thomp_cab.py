"""
Context aware clustering of bandits based on Thompson sampling
"""

import logging
import time
import numpy as np
import multiprocessing as mp
from itertools import repeat

LOGGER = logging.getLogger(__name__)


class ThompCAB(object):
    r"""
    Context aware clustering of bandits using Thompson Sampling

    Parameters: todo

    References: todo

    """

    def __init__(self, users, d=128, gamma=0.5, minUsed=1,
                p=0.2, delta=0.5, R=0.01, epsilon=0.5):

        self.numUsers = users # number of Users
        self.d = d
        self.minUsed = minUsed
        self.gamma = gamma
        self.p = p
        self.t = 0
        # self.used = np.zeros((self.numUsers),dtype=int) # for user inverse
        # self.used = [0] * self.numUsers
        self.used = np.zeros(self.numUsers)
        self.updated = [0] * self.numUsers
        # Thompson sampling parameters
        self.v = R * np.sqrt(24 / epsilon
                             * self.d
                             * np.log(1 / delta))
        self._init_action_model()


    def _init_action_model(self):
        """
        """
        self.mu_hat = np.zeros(shape=(self.numUsers,self.d))
        self.mu_tilde = np.zeros(shape=(self.numUsers,self.d))
        self.f = np.zeros(shape=(self.numUsers,self.d))
        self.B_inv = np.zeros(shape=(self.numUsers,self.d,self.d))
        for i, mat in enumerate(self.B_inv):
            self.B_inv[i] = np.eye(self.d)


    def _compute_neighboroods(self, context_array, user):
        """

        """
        K = len(context_array)
        # define neighborood sets and 'confidence bounds'
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)
        self.CB = np.zeros(shape=(self.numUsers, K))

        # compute user parameters
        self.mu_tilde[user] = np.random.multivariate_normal(
            self.mu_hat[user].flat, self.v**2 * self.B_inv[user]) #[np.newaxis, ...]
        user_estimated_reward_array = context_array.dot(self.mu_hat[user])
        user_score_array = context_array.dot(self.mu_tilde[user])
        self.CB[user] = np.abs(user_score_array - user_estimated_reward_array)

        # iterate over users
        for j in range(self.numUsers):
            # if user, go to next step
            if j == user:
                self.N[user] = 1
            elif self.used[j] >= self.minUsed and np.random.rand() <= self.p:
                # sample mu tilde
                if self.updated[j]:
                    self.mu_tilde[j] = np.random.multivariate_normal(
                        self.mu_hat[j].flat, self.v**2 * self.B_inv[j])

                # compute scores
                estimated_reward_array = context_array.dot(self.mu_hat[j])
                score_array = context_array.dot(self.mu_tilde[j])
                # compute "confidence bound" and update neighborood
                self.CB[j] = np.abs(score_array - estimated_reward_array)
                # update N
                one = np.abs(estimated_reward_array - user_estimated_reward_array)
                two = self.CB[user] + self.CB[j]
                a = np.where(one<two)[0]
                self.N[j,a] = 1

        # pool = mp.Pool(processes=mp.cpu_count())
        # # N = pool.starmap(self._score, zip(repeat(context_array), range(self.numUsers), 
        # #                     user_estimated_reward_array))
        # N = [pool.apply(self._score, args=(context_array, range(self.numUsers),
        #         user_estimated_reward_array))]
        # print(len(N[0]))

        # results = [pool.apply(cube, args=(x,)) for x in range(1,7)]


    def _compute_neigh_vectorized(self, context_array, user):
        """
        context_array: row vector
        user: user_id
        """
        K = len(context_array)
        # define neighborood sets
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)
        # self.CB = np.zeros(shape=(K, self.numUsers))

        # sample mu tilde
        for j in range(self.numUsers):
            if self.used[j] >= self.minUsed and self.updated[j]:
                    self.mu_tilde[j] = np.random.multivariate_normal(
                            self.mu_hat[j].flat, self.v**2 * self.B_inv[j])

        # self.mu_tilde[user] = np.random.multivariate_normal(
        #                     self.mu_hat[user].flat, self.v**2 * self.B_inv[user])

        """
        # # using indices for users
        # inds = np.where(self.used)[0]
        # u = np.where(inds==user)[0][0]
        # estimated_reward = context_array.dot(self.mu_hat[inds].T)
        # payoff = context_array.dot(self.mu_tilde[inds].T)
        # self.CB[:, inds] = np.abs(payoff - estimated_reward)
        # i, j = np.where( np.abs(estimated_reward - estimated_reward[:,u][..., np.newaxis]) < 
        #                 self.CB[:, u][..., np.newaxis] + self.CB[:, inds] )

        # self.N[inds[j], i] = 1
        # self.CB = self.CB.T
        """

        estimated_reward = self.mu_hat.dot(context_array.T)
        payoff = self.mu_tilde.dot(context_array.T)
        self.CB = np.abs(payoff - estimated_reward)
        i, j = np.where( np.abs(estimated_reward - estimated_reward[user]) < 
            self.CB[user] + self.CB)
        self.N[i, j] = 1
        # set users never used to 0
        t = np.where(self.used == 0)[0]
        self.N[t,:] = 0

        """
        # working (?) version
        estimated_reward = context_array.dot(self.mu_hat.T)
        payoff = context_array.dot(self.mu_tilde.T)
        self.CB = np.abs(payoff - estimated_reward)
        i, j = np.where( np.abs(estimated_reward - estimated_reward[:,user][..., np.newaxis]) <
                        self.CB[:, user][..., np.newaxis] + self.CB )
        self.N[j, i] = 1
        self.CB = self.CB.T
        """


    def get_action(self, context_array, user):
        """
        """
        self.t += 1
        # update parameters in order to sample user's mu tilde
        self.used[user] += 1
        self.updated[user] = 1
        # self._compute_neighboroods(context_array, user)
        self._compute_neigh_vectorized(context_array, user)
        avg_mu_tilde = self.mu_tilde.T.dot(self.N) / np.sum(self.N, axis=0)
        payoff = np.sum(np.multiply(context_array, avg_mu_tilde.T), axis=1) # is it right?
        return np.argmax(payoff)


    def reward(self, x, reward, user, action_id):
        """
        """
        self.updated = [0] * self.numUsers
        self._update(user, x, reward)
        if self.CB[user, action_id] <= self.gamma/4:
            # print(np.sum(self.N[:,action_id]))
            for j in self.N[:,action_id]:
                if j == user:
                    continue
                elif self.CB[j, action_id] <= self.gamma/4:
                    self._update(j, x, reward)


    def _update(self, user, x, reward):
        """
        Update the model.

        Input: 
            - model: (dict) containing all models
            - user:  (int) id of the user
            - x:     (np.array) context of the action performed
        """
        self.updated[user] = 1 # update to sample at next step
        self.f[user] += reward * x
        B_inv = sherman_morrison(self.B_inv[user], x)
        self.B_inv[user] = B_inv
        self.mu_hat[user] = B_inv.dot(self.f[user])


def sherman_morrison(M_inv, x):
    """
    Input:
        - x: (np.array) column vector
        - M_inv: (np.array) inverse of M matrix
    Output:
        (M + x*x')^-1 computed using Sherman-Morrison formula
    """
    return M_inv - M_inv.dot(x.dot(x.T.dot(M_inv)))/(1+x.T.dot(M_inv.dot(x)))











