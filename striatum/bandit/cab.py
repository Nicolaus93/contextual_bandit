"""
Context aware clustering of bandits
"""

import logging
import numpy as np
from .bandit import BaseBandit
from collections import defaultdict
import copy
import time
import six

LOGGER = logging.getLogger(__name__)

class CAB(BaseBandit):
    r"""
    Context aware clustering of bandits

    Parameters:


    References:
    ----------
    .. [1] 

    """

    def __init__(self, history_storage, model_storage, action_storage, users,
                recommendation_cls=None, context_dimension=128, gamma=0.5, alpha=0.5, 
                minUsed=0, p=0.2):

        super(CAB, self).__init__(history_storage, model_storage,
                                    action_storage, recommendation_cls)

        self.numUsers = users # number of Users
        # self.users = {} # dict for users
        self.context_dimension = context_dimension
        self.alpha = alpha
        self.minUsed = minUsed
        self.gamma = gamma
        self.p = p
        self.t = 0
        self.used = np.zeros((self.numUsers),dtype=int) # for user inverse

        # Initialize LinUCB Model Parameters
        model = {
            # dictionary - For any action a in actions,
            # A[a] = (DaT*Da + I) the ridge reg solution
            'A': {},
            # dictionary - The inverse of each A[a] for action a
            # in actions
            'A_inv': {},
            # dictionary - The cumulative return of action a, given the
            # context xt.
            'b': {},
            # dictionary - The coefficient vector of actions with
            # linear model b = dot(xt, theta)
            'theta': {},
            # dictionary - 
            # 
            'CB': {}
        }

        models = {}
        # for action_id in self._action_storage.iterids():
        self._init_action_model(model, models)

        self._model_storage.save_model(models)

    def _init_action_model(self, model, user_models):
        """
        initiate model for each user
        """
        model['A'] = np.identity(self.context_dimension)
        model['A_inv'] = np.identity(self.context_dimension)
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))
        for i in range(self.numUsers):
            user_models[i] = copy.deepcopy(model)

    def _cab_score(self, context, user):
        """Context aware clustering of bandits
        """
        model = self._model_storage.get_model()
        user_A_inv = model[user]['A_inv']
        user_theta = model[user]['theta']

        CB = {}
        for j in range(self.numUsers):
            CB[j] = {}
            for action_id in self._action_storage:
                CB[j][action_id] = 0

        # compute neighbourood sets
        self.N = defaultdict(list)
        for action_id in self._action_storage:
            action_context = np.reshape(context[action_id], (-1, 1))
            CB[user][action_id] = float(self.alpha * np.sqrt(action_context.T
                             .dot(user_A_inv)
                             .dot(action_context))) * np.log(self.t+1)
            for j in range(self.numUsers):
                if j == user:
                    self.N[action_id].append(j)
                    continue
                elif self.used[j] >= self.minUsed and np.random.rand() <= self.p:
                    A_inv = model[j]['A_inv']
                    theta = model[j]['theta']
                    CB[j][action_id] = float(self.alpha * np.sqrt(action_context.T
                                 .dot(user_A_inv)
                                 .dot(action_context))) * np.log(self.t+1)
                    if np.abs(action_context.T.dot(user_theta-theta)) <= CB[user][action_id]+CB[j][action_id]:
                        self.N[action_id].append(j)

        # compute payoffs
        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id in self._action_storage:
            action_context = np.reshape(context[action_id], (-1, 1))
            theta_sum = np.zeros((self.context_dimension,1))
            CB_ag = 0
            for j in self.N[action_id]:
                # print(len(self.N[action_id])) # debugging
                theta_sum = theta_sum + model[j]['theta']
                CB_ag = CB_ag + CB[j][action_id]

            avg_theta = theta_sum / len(self.N[action_id])
            avg_CB = CB_ag / len(self.N[action_id])
            estimated_reward[action_id] = float(action_context.T.dot(avg_theta))
            uncertainty[action_id] = float(avg_CB)
            score[action_id] = estimated_reward[action_id] + uncertainty[action_id]
 
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
            Id of the user.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """

        if not isinstance(context, dict):
            raise ValueError("Cab requires context dict for all actions!")
        if n_actions == -1:
            n_actions = self.action_storage.count()

        estimated_reward, uncertainty, score = self._cab_score(context, user)

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

        # update some parameters
        self.t += 1
        self.used[user] += 1

        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    def reward(self, history_id, rewards, user):
        """
        """
        context = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .context)

        recommendations = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .recommendations)

        # Update the model
        model = self._model_storage.get_model()

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1,1))
            CB = float(self.alpha * np.sqrt(action_context.T
                                     .dot(model[user]['A_inv'])
                                     .dot(action_context))) * np.log(self.t+1)
            # print("CB: " + str(CB))
            # print("threshold: " + str(self.alpha * self.gamma/4 * np.log(self.t+1)))

            if CB > self.alpha * self.gamma/4 * np.log(self.t+1):
                # print("alone")
                action_context = np.reshape(context[action_id], (-1,1))
                model[user]['A'] += action_context.dot(action_context.T)
                model[user]['A_inv'] = np.linalg.inv(model[user]['A'])
                model[user]['b'] += reward * action_context
                model[user]['theta'] = model[user]['A_inv'].dot(model[user]['b'])
            else:
                for action_id, reward in six.viewitems(rewards):
                    for j in self.N[action_id]:
                        CB_j = float(self.alpha * np.sqrt(action_context.T
                                         .dot(model[j]['A_inv'])
                                         .dot(action_context))) * np.log(self.t+1)
                        if CB_j <= self.alpha * self.gamma/4 * np.log(self.t+1):
                            print(len(self.N[action_id]))
                            action_context = np.reshape(context[action_id], (-1,1))
                            model[j]['A'] += action_context.dot(action_context.T)
                            model[j]['A_inv'] = np.linalg.inv(model[j]['A'])
                            model[j]['b'] += reward * action_context
                            model[j]['theta'] = model[j]['A_inv'].dot(model[j]['b'])

            self._model_storage.save_model(model)

        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def get_parameters(self, actions):
        """
        """


    def add_action(self, actions):
        """
        """
        return

    def remove_action(self, actions):
        """
        """
        return








