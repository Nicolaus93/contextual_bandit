import numpy as np
import random
from .utils import get_random_state, sherman_morrison
from numpy.linalg import inv


class ThompsonOne(object):
    r"""

    Parameters: todo

    References: todo

    """

    def __init__(self, d=128, delta=0.5, R=0.01, 
                    epsilon=0.5, random_state=None):

        self.d = d
        self.random_state = get_random_state(random_state)
        # Thompson sampling parameters
        self.v = R * np.sqrt(24 / epsilon
                             * self.d
                             * np.log(1 / delta))
        self.v = 0.1
        self._init_model()


    def _init_model(self):
        """
        """
        self.mu_hat = np.zeros(shape=(self.d))
        self.f = np.zeros(shape=(self.d))
        self.B_inv = np.eye(self.d)


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
        """
        mu_tilde = self.random_state.multivariate_normal(self.mu_hat.flat, self.v**2 * self.B_inv)
        payoff = mu_tilde.dot(context_array.T)
        action_id = np.argmax(payoff)
        return action_id


    def reward(self, x, reward, action_id):
        """
        """
        self.f += reward * x
        self.B_inv = sherman_morrison(self.B_inv, x)
        self.mu_hat = self.B_inv.dot(self.f)

