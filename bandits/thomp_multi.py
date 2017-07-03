import numpy as np
import random
from .utils import get_random_state, sherman_morrison
from numpy.linalg import inv


class ThompMulti(object):
    r"""

    Parameters: todo

    References: todo

    """

    def __init__(self, users, d=128, delta=0.5, R=0.01, 
                    epsilon=0.5, random_state=None):

        self.numUsers = users # number of Users
        self.d = d
        self.random_state = get_random_state(random_state)

        # Thompson sampling parameters
        self.v = R * np.sqrt(24 / epsilon
                             * self.d
                             * np.log(1 / delta))
        self.v = 0.1
        # print(self.v)
        self._init_model()


    def _init_model(self):
        """
        """
        self.mu_hat = np.zeros(shape=(self.numUsers,self.d))
        self.mu_tilde = np.zeros(shape=(self.numUsers,self.d))
        self.f = np.zeros(shape=(self.numUsers,self.d))
        ###
        self.B = np.zeros(shape=(self.numUsers,self.d,self.d))
        ###
        self.B_inv = np.zeros(shape=(self.numUsers,self.d,self.d))
        for i, mat in enumerate(self.B_inv):
            ###
            self.B[i] = np.eye(self.d)
            ###
            self.B_inv[i] = np.eye(self.d)

    def get_action(self, context_array, user):
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
        self.mu_tilde[user] = self.random_state.multivariate_normal(self.mu_hat[user].flat, self.v**2 * self.B_inv[user])
        payoff = self.mu_tilde[user].dot(context_array.T)
        action_id = np.argmax(payoff)
        return action_id
        # best_actions = np.argwhere(payoff == np.amax(payoff))
        # if len(best_actions) > 1:
        #     b = list(np.squeeze(best_actions))
        #     best = random.sample(b,1)[0]
        # else:
        #     best = np.squeeze(best_actions)
        # self.aaa.append(best)
        # return best


    def reward(self, x, reward, user, action_id):
        """
        """
        self.f[user] += reward * x
        
        B_inv = sherman_morrison(self.B_inv[user], x)
        # x = np.reshape(x, (-1, 1))
        # self.B += x.dot(x.T)  # pylint: disable=invalid-name
        # B_inv = inv(self.B[user])

        self.B_inv[user] = B_inv
        self.mu_hat[user] = B_inv.dot(self.f[user])

