""" Thompson Sampling with Linear Payoff
This module contains a class that implements Thompson Sampling with Linear
Payoff. Thompson Sampling with linear payoff is a contexutal multi-armed bandit
algorithm which assume the underlying relationship between rewards and contexts
is linear. The sampling method is used to balance the exploration and
exploitation. Please check the reference for more details.
"""
import numpy as np
from .utils import get_random_state, sherman_morrison


class ThompsonOne(object):
    r"""
    d : int
        dimension of the context arrays

    v : float
        parameter controlling the variance

    random_state: {int, np.random.RandomState} (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    References
    ----------
    .. [1]  Shipra Agrawal, and Navin Goyal. "Thompson Sampling for Contextual
            Bandits with Linear Payoffs." Advances in Neural Information
            Processing Systems 24. 2011.
    """

    def __init__(self, d=128, random_state=None, v=0):
        self.d = d
        self.random_state = get_random_state(random_state)
        self.v = v
        self._init_model()

    def _init_model(self):
        self.mu_hat = np.zeros(shape=(self.d))
        self.f = np.zeros(shape=(self.d))
        self.B_inv = np.eye(self.d)

    # @profile
    def get_action(self, context_array):
        """Return the action to perform

        Parameters
        ----------
        context_array : numpy array
            K x d array containing contexts for every action.

        Returns
        -------
        action_id : int
            Id of the action that will be recommended to the user.
        """
        mu_tilde = self.random_state.multivariate_normal(
            self.mu_hat.flat, self.v**2 * self.B_inv)
        payoff = mu_tilde.dot(context_array.T)
        action_id = np.argmax(payoff)
        return action_id

    # @profile
    def reward(self, x, reward, action_id):
        """Reward the previous action with reward.

        Parameters
        ----------
        x : int
            The history id of the action to reward.

        reward : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        self.f += reward * x
        self.B_inv = sherman_morrison(self.B_inv, x)
        self.mu_hat = self.B_inv.dot(self.f)
