"""
Context aware clustering of bandits based on Thompson sampling
"""

import numpy as np
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
                 v=0.1, random_state=None, p=1):

        self.numUsers = users
        self.d = d
        self.random_state = get_random_state(random_state)
        self.gamma = gamma
        self.t = 0
        self.p = p
        self.used = np.zeros(self.numUsers)
        self.updated = np.zeros(self.numUsers, dtype=int)
        self.neigh_size = []
        self.updated_size = []
        # Thompson sampling parameters
        self.v = v
        self._init_model()

    def _init_model(self):
        # self.mu_hat = np.zeros(shape=(self.numUsers, self.d))
        self.mu_hat = self.random_state.multivariate_normal(
            np.zeros(self.d), 0.001 * np.identity(self.d), size=self.numUsers)
        self.mu_tilde = np.zeros(shape=(self.numUsers, self.d))
        self.f = np.zeros(shape=(self.numUsers, self.d))
        self.B_inv = np.zeros(shape=(self.numUsers, self.d, self.d))
        for i, mat in enumerate(self.B_inv):
            self.B_inv[i] = np.eye(self.d)

    # @profile
    def _compute_neigh_vectorized(self, context_array, user):
        """
        context_array: row vector
        user: user_id
        """
        K = len(context_array)
        # define neighborood sets
        self.N = np.zeros(shape=(self.numUsers, K), dtype=int)

        remove = np.zeros(self.numUsers, dtype=int)

        # sample mu tilde
        for j in np.where(self.updated)[0]:
            if np.random.rand() < self.p:
                self.mu_tilde[j] = self.random_state.multivariate_normal(
                    self.mu_hat[j].flat, self.v**2 * self.B_inv[j])
            else:
                remove[j] = 1

        remove[user] = 0

        estimated_reward = self.mu_hat.dot(context_array.T)
        estimated_reward[np.where(remove)[0], :] = np.inf
        payoff = self.mu_tilde.dot(context_array.T)
        self.CB = np.abs(payoff - estimated_reward)

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

        # update parameters in order to sample user's mu tilde
        self.used[user] += 1
        # self.updated.add(user)
        self.updated[user] = 1
        # compute neighbourhood
        self._compute_neigh_vectorized(context_array, user)
        avg_mu_tilde = self.mu_tilde.T.dot(self.N) / np.sum(self.N, axis=0)
        payoff = np.sum(np.multiply(context_array, avg_mu_tilde.T),
                        axis=1)  # is it right?
        # payoff = self.mu_tilde[user].dot(context_array.T)
        action_id = np.argmax(payoff)
        return action_id

    def reward(self, x, reward, action_id):
        """
        Update the model
        """
        user = self.user
        self.updated = np.zeros(self.numUsers, dtype=int)
        self._update(user, x, reward)  # update the user model first
        # update neighborhood size
        self.neigh_size.append(sum(self.N[:, action_id]))

        # # update users in the neighborhood
        if self.CB[user, action_id] <= self.gamma:
            self.N[user, action_id] = 0
            to_update = self.CB[:, action_id] * self.N[:, action_id]
            for j, value in enumerate(to_update):
                if value > 0:
                    if value <= self.gamma / self.used[j] and value > 0:
                    # if value > 0 and value <= self.gamma:
                        self._update(j, x, reward)

        # update updated size
        # print(sum(self.updated))
        self.updated_size.append(sum(self.updated))

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
        self.updated[user] = 1
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
