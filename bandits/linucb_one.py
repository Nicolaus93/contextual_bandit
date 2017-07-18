import numpy as np
from .utils import sherman_morrison


class LinUcbOne(object):
    r"""LinUCB with Disjoint Linear Models

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

    def __init__(self, d=128, alpha=0.5):
        self.d = d
        self.alpha = alpha
        self._init_model()

    def _init_model(self):
        """
        """
        self.theta = np.zeros(shape=(self.d))
        self.b = np.zeros(shape=(self.d))
        self.A_inv = np.eye(self.d)

    def get_action(self, context_array):
        """
        """
        estimated_reward = self.theta.dot(context_array.T)
        uncertainty = np.zeros(shape=context_array.shape[0])
        for i, action in enumerate(context_array):
            action_context = np.reshape(context_array[i], (-1, 1))
            uncertainty[i] = self.alpha * np.sqrt(action_context.T
                                                  .dot(self.A_inv)
                                                  .dot(action_context))
        payoff = estimated_reward + uncertainty
        action_id = np.argmax(payoff)
        return action_id

    def reward(self, x, reward, action_id):
        """
        """
        self.b += reward * x
        x = x.reshape((-1, 1))
        self.A_inv = sherman_morrison(self.A_inv, x)
        self.theta = self.A_inv.dot(self.b)
