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
from sklearn.metrics.pairwise import cosine_similarity

from .bandit import BaseBandit
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)

class ThompCAB(BaseBandit):
    r"""
    Context aware clustering of bandits using Thompson Sampling

    Parameters: todo

    References: todo

    """

    def __init__(self, history_storage, model_storage, action_storage, users,
                recommendation_cls=None, context_dimension=128, gamma=0.5, alpha=0.5,
                minUsed=0, p=0.2, random_state=None, delta=0.5, R=0.01, epsilon=0.5):

        super(ThompCAB, self).__init__(history_storage, model_storage,
                                    action_storage, recommendation_cls)

        self.numUsers = users # number of Users
        self.context_dimension = context_dimension
        self.alpha = alpha
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

        for j in range(self.numUsers):
            if j == user:
                for action_id in action_ids:
                    self.N[action_id].append(j)
                continue
            elif self.used[j] >= self.minUsed:
                # B = model[j]['B']
                mu_hat = model[j]['mu_hat']
                # mu_tilde = self.random_state.multivariate_normal(
                #     mu_hat.flat, v**2 * np.linalg.inv(B))[..., np.newaxis]
                estimated_reward_array = context_array.dot(mu_hat)
                # score_array = context_array.dot(mu_tilde)
                # for action_id, estimated_reward, score in zip(
                #         action_ids, estimated_reward_array, score_array):
                for action_id, estimated_reward in zip(
                    action_ids, estimated_reward_array):
                    r = np.random.rand()
                    if r <= self.p:
                        # CHANGE HERE!
                        # if cosine_similarity(user_mu_tilde, mu_hat) > self.gamma:
                        if cosine_similarity(user_mu_hat.reshape(1,-1), mu_hat.reshape(1,-1)) > self.gamma:
                            self.N[action_id].append(j)
                        # if np.abs(user_score_array-score_array) <= ...
                        #     self.N[action_id].append(j)

        # compute payoffs
        estimated_reward = {}
        uncertainty = {}
        score = {}

        for action_id in self._action_storage:
            action_context = np.reshape(context[action_id], (-1,1))
            mu_hat_sum = np.zeros(shape=(self.context_dimension, 1))
            B_sum = np.zeros(shape=(self.context_dimension, self.context_dimension))

            for j in self.N[action_id]:
                B = model[j]['B']
                mu_hat = model[j]['mu_hat']
                # mu_tilde = self.random_state.multivariate_normal(
                #         mu_hat.flat, v**2 * np.linalg.inv(B))[..., np.newaxis]

                mu_hat_sum += mu_hat
                B_sum += B
                # mu_tilde_sum += mu_tilde

            avg_mu_hat = mu_hat_sum / len(self.N[action_id])
            avg_B = B_sum / len(self.N[action_id])
            # sample this (check whether the variance is good!)
            avg_mu_tilde = self.random_state.multivariate_normal(
                avg_mu_hat.flat, v**2 * np.linalg.inv(avg_B))[..., np.newaxis]
            estimated_reward[action_id] = float(action_context.T.dot(avg_mu_hat))
            score[action_id] = float(action_context.T.dot(avg_mu_tilde))
            uncertainty[action_id] = float(score[action_id] - estimated_reward[action_id])

        return estimated_reward, uncertainty, score


    def get_action(self, context, user, n_actions=None):
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
                a = score[action_id]
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
            # WARNING! USE ONLY WITH MOVIELENS (1 ACTION REQUIRED), OTHERWISE FIX IT
            user_estimated_reward = recommendations[0].estimated_reward
            user_score = recommendations[0].score
            CB = float(user_score - user_estimated_reward)

            print("Thompson: " + str(CB))
            print("threshold: " + str(self.alpha * self.gamma/4 * np.log(self.t+1)))
            
            if CB > self.alpha * self.gamma/4 * np.log(self.t+1):
                print("alone")
                context_t = np.reshape(context[action_id], (-1, 1))
                model[user]['B'] += context_t.dot(context_t.T)  # pylint: disable=invalid-name
                model[user]['f'] += reward * context_t
                model[user]['mu_hat'] = np.linalg.inv(model[user]['B']).dot(model[user]['f'])
            else:
                for action_id, reward in six.viewitems(rewards):
                    for j in self.N[action_id]:
                        # what here?
                        CB_j = 1

                        if CB_j <= self.alpha * self.gamma/4 * np.log(self.t+1):
                            context_t = np.reshape(context[action_id], (-1, 1))
                            model[j]['B'] += context_t.dot(context_t.T)  # pylint: disable=invalid-name
                            model[j]['f'] += reward * context_t
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








