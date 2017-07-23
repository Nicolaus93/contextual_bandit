import numpy as np
from .utils import get_random_state, sherman_morrison


class ThompMulti(object):
    r"""
    d : int
        dimension of the context arrays

    users : int
        number of users

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

    def __init__(self, users, d=128, random_state=None, v=0):

        self.numUsers = users  # number of Users
        self.d = d
        self.random_state = get_random_state(random_state)
        # Thompson sampling parameters
        self.v = v
        self._init_model()

    def _init_model(self):
        self.mu_hat = np.zeros(shape=(self.numUsers, self.d))
        self.f = np.zeros(shape=(self.numUsers, self.d))
        self.B_inv = np.zeros(shape=(self.numUsers, self.d, self.d))
        for i, mat in enumerate(self.B_inv):
            self.B_inv[i] = np.eye(self.d)

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
        user = self.user
        self.mu_tilde = np.random.multivariate_normal(
            self.mu_hat[user].flat, self.v**2 * self.B_inv[user])
        payoff = self.mu_tilde.dot(context_array.T)
        action_id = np.argmax(payoff)
        return action_id

    def reward(self, x, reward, action_id):
        """
        Update the model after receiving reward.

        Parameters
        ----------
        user : int
            id of the user

        x : np.array
            context of the action performed

        reward : float or int
            reward for the action taken

        action_id : int
            id of the action in the current context
        """
        user = self.user
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
