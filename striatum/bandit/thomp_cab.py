"""
Context aware clustering of bandits based on Thompson sampling
"""

import logging
import copy
import time
import six
from six.moves import zip
from collections import defaultdict

import numpy as np

from .bandit import BaseBandit
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)

def sherman_morrison(M_inv, x):
    """
    Input:
        - x, column vector
        - M_inv, inverse of M matrix
    Output:
        (M + x*x')^-1 computed using Sherman-Morrison formula
    """
    return M_inv - M_inv.dot(x.dot(x.T.dot(M_inv)))/(1+x.T.dot(M_inv.dot(x)))


class ThompCAB(BaseBandit):
    r"""
    Context aware clustering of bandits using Thompson Sampling

    Parameters: todo

    References: todo

    """

    def __init__(self, history_storage, model_storage, action_storage, users,
                recommendation_cls=None, context_dimension=128, gamma=0.5,
                minUsed=0, p=0.2, random_state=None, delta=0.5, R=0.01, epsilon=0.5):

        super(ThompCAB, self).__init__(history_storage, model_storage,
                                    action_storage, recommendation_cls)

        self.numUsers = users # number of Users
        self.context_dimension = context_dimension
        # self.alpha = alpha
        self.minUsed = minUsed
        self.gamma = gamma
        self.p = p
        self.t = 0
        self.used = np.zeros((self.numUsers),dtype=int) # for user inverse
        # Thompson sampling parameters
        self.random_state = get_random_state(random_state)
        self.R = R
        self.delta = delta
        self.epsilon = epsilon

        # Initialize Model Parameters
        model = {
            # dictionary - 
            'B': {},
            # dictionary - 
            'mu_hat': {},
            # dictionary - 
            'f': {},
        }

        # model initialization
        models = self._init_action_model(model)
        self._model_storage.save_model(models)

    def _init_action_model(self, model):
        """
        Initiate model for each user
        """
        user_models = {}
        model['B'] = np.identity(self.context_dimension)
        model['mu_hat'] = np.zeros(shape=(self.context_dimension, 1))
        model['f'] = np.zeros(shape=(self.context_dimension, 1))
        for i in range(self.numUsers):
            user_models[i] = copy.deepcopy(model)
        return user_models

    def _thomp_cab_score(self, context, user):
        """
        Thompson Sampling for CAB algorithm
        """
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id]
                                    for action_id in action_ids])
        model = self._model_storage.get_model()
        # v parameter
        v = self.R * np.sqrt(24 / self.epsilon
                             * self.context_dimension
                             * np.log(1 / self.delta))
        # compute neighbourood sets
        self.N = defaultdict(list)
        user_B = model[user]['B']
        user_mu_hat = model[user]['mu_hat']
        user_mu_tilde = self.random_state.multivariate_normal(
            user_mu_hat.flat, v**2 * np.linalg.inv(user_B))[..., np.newaxis]
        user_estimated_reward_array = context_array.dot(user_mu_hat)
        user_score_array = context_array.dot(user_mu_tilde)
        user_CB = user_score_array - user_estimated_reward_array
        # store mu tilde for all users
        self.mu_tilde_array = [0] * self.numUsers

        # iterate over users
        for j in range(self.numUsers):
            # if user, go to next step
            if j == user:
                for action_id in action_ids:
                    self.N[action_id].append(j)
                    self.mu_tilde_array[j] = user_mu_tilde
                continue
            elif self.used[j] >= self.minUsed and np.random.rand() <= self.p:
                # retrieve j-th user parameters and sample mu tilde
                B = model[j]['B']
                mu_hat = model[j]['mu_hat']
                mu_tilde = self.random_state.multivariate_normal(
                    mu_hat.flat, v**2 * np.linalg.inv(B))[..., np.newaxis]
                # store mu tilde for later
                self.mu_tilde_array[j] = mu_tilde
                # compute scores
                estimated_reward_array = context_array.dot(mu_hat)
                score_array = context_array.dot(mu_tilde)
                # compute "confidence bound" and update neighborood
                j_CB = score_array - estimated_reward_array
                one = np.abs(estimated_reward_array - user_estimated_reward_array)
                two = user_CB + j_CB
                a = np.where(one<two)[0]
                for i in a:
                    self.N[action_ids[i]].append(j)

        # compute payoffs
        estimated_reward = {}
        uncertainty = {}
        score = {}

        for action_id in self._action_storage:  # is this correct (action_id in self._action_storage) ?
            action_context = np.reshape(context[action_id], (-1,1))
            mu_tilde_sum = np.zeros(shape=(self.context_dimension,1))
            mu_hat_sum = np.zeros(shape=(self.context_dimension, 1))

            for j in self.N[action_id]:
                mu_hat = model[j]['mu_hat']
                mu_hat_sum += mu_hat
                mu_tilde_sum += self.mu_tilde_array[j]

            avg_mu_hat = mu_hat_sum / len(self.N[action_id])
            avg_mu_tilde = mu_tilde_sum / len(self.N[action_id])
            estimated_reward[action_id] = float(action_context.T.dot(avg_mu_hat))
            score[action_id] = float(action_context.T.dot(avg_mu_tilde))
            uncertainty[action_id] = float(score[action_id] - estimated_reward[action_id])

        return estimated_reward, uncertainty, score


    def get_action(self, context, user, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : dict
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        user: int 
            Id of the user

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if not isinstance(context, dict):
            raise ValueError("Thompson cab requires context dict for all actions!")
        if n_actions == -1:
            n_actions = self.action_storage.count()

        estimated_reward, uncertainty, score = self._thomp_cab_score(context, user)

        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            recommendations = self._recommendation_cls(
                action=1,
                estimated_reward=estimated_reward[1],
                uncertainty=uncertainty[1],
                score=0
                )
        else:
            recommendation_ids = sorted(score, key=score.get, 
                                        reverse=True)[:n_actions]
            recommendations = []
            for action_id in recommendation_ids:
                recommendations.append(self._recommendation_cls(
                    action=action_id,
                    # action=self._action_storage.get(action_id),
                    estimated_reward=estimated_reward[action_id],
                    uncertainty=uncertainty[action_id],
                    score=score[action_id],
                ))

        self.t += 1
        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations


    def reward(self, history_id, rewards, user):
        """
        """
        context = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .context)
        # list of recommedations
        recommendations = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .recommendations)

        # Update the model
        model = self._model_storage.get_model()

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1,1))
            user_estimated_reward = action_context.T.dot(model[user]['mu_hat'])
            user_score = action_context.T.dot(self.mu_tilde_array[user])
            user_CB = user_score - user_estimated_reward

            # if user_CB > self.alpha * self.gamma/4 * np.log(self.t+1):
            if user_CB > self.gamma/4 * np.log(self.t+1):
                # print("alone")
                model[user]['B'] += action_context.dot(action_context.T)
                model[user]['f'] += reward * action_context
                model[user]['mu_hat'] = np.linalg.inv(model[user]['B']).dot(model[user]['f'])
            else:
                for action_id, reward in six.viewitems(rewards):
                    for j in self.N[action_id]:
                        mu_hat = model[j]['mu_hat']
                        mu_tilde = self.mu_tilde_array[j]
                        estimated_reward = action_context.T.dot(mu_hat)
                        score = action_context.T.dot(mu_tilde)
                        j_CB = float(score - estimated_reward)
                        # update model
                        if j_CB <= self.gamma/4 * np.log(self.t+1):
                        # if j_CB <= self.alpha * self.gamma/4 * np.log(self.t+1):
                            model[j]['B'] += action_context.dot(action_context.T)
                            model[j]['f'] += reward * action_context
                            model[j]['mu_hat'] = np.linalg.inv(model[j]['B']).dot(model[j]['f'])

            self._model_storage.save_model(model)

        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """
        """
        return

    def remove_action(self, actions):
        """
        """
        return








