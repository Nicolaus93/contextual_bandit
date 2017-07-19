import numpy as np
from .utils import sherman_morrison


class LinUcbMulti(object):
    r"""LinUCB with Disjoint Linear Models for individual users.

    Parameters
    ----------
    context_dimension: int
        The dimension of the context.

    alpha: float
        The constant determines the width of the upper confidence bound.

    References
    ----------
    .. [1]  Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
            News Article Recommendation." In Proceedings of the 19th
            International Conference on World Wide Web (WWW), 2010.
    """

    def __init__(self, users, d=128, alpha=0.5):
        self.d = d
        self.alpha = alpha
        self._init_model(users)

    def _init_model(self, numUsers):
        self.theta = np.zeros(shape=(numUsers, self.d))
        self.b = np.zeros(shape=(numUsers, self.d))
        self.A_inv = np.zeros(shape=(numUsers, self.d, self.d))
        for i, mat in enumerate(self.A_inv):
            self.A_inv[i] = np.eye(self.d)

    def get_action(self, context_array, user):
        """
        """
        estimated_reward = self.theta[user].dot(context_array.T)
        uncertainty = np.zeros(shape=context_array.shape[0])
        for i, action in enumerate(context_array):
            action_context = np.reshape(context_array[i], (-1, 1))
            uncertainty[i] = self.alpha * np.sqrt(action_context.T
                                                  .dot(self.A_inv[user])
                                                  .dot(action_context))
        payoff = estimated_reward + uncertainty
        action_id = np.argmax(payoff)
        return action_id

    def reward(self, x, reward, action_id, user):
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

        action_id : int
        """
        self.b[user] += reward * x
        self.A_inv[user] = sherman_morrison(self.A_inv[user], x)
        self.theta[user] = self.A_inv[user].dot(self.b[user])
