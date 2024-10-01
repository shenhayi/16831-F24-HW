import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log

        # raise NotImplementedError
        # Calculate Q-values
        q_values = self.calculate_q_vals(rewards_list)

        # Estimate advantages
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)

        # Update the policy using the actor's update method
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log


    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        q_values = []
        if not self.reward_to_go:
            # Use the whole trajectory for each timestep
            for rewards in rewards_list:
                discounted_return = self._discounted_return(rewards)
                q_values.extend([discounted_return] * len(rewards))
        else:
            # Use reward-to-go for each timestep
            for rewards in rewards_list:
                discounted_cumsums = self._discounted_cumsum(rewards)
                q_values.extend(discounted_cumsums)
        
        return np.array(q_values)

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            # Predict the baseline values using the actor's value function
            values_normalized = self.actor.run_baseline_prediction(obs)
            assert values_normalized.ndim == q_values.ndim

            # Unnormalize the value predictions
            values = unnormalize(values_normalized, q_values)

            if self.gae_lambda is not None:
                # Initialize advantages array with an extra element for the dummy value
                values = np.append(values, [0])
                rewards = np.concatenate(rewards_list)
                batch_size = len(rewards)
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    if terminals[i]:
                        delta = rewards[i] - values[i]
                        advantages[i] = delta
                    else:
                        delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1]
                
                # Remove the dummy advantage
                advantages = advantages[:-1]
            else:
                # Compute advantages as Q-values minus baseline values
                advantages = q_values - values

            # Update the baseline by fitting to Q-values
            self.actor.fit_baseline(obs, q_values)

        else:
            # If no baseline, the advantage is just the Q-values
            advantages = q_values.copy()

        # Normalize the advantages if required
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create discounted_returns
        discounted_returns = 0
        for t, r in enumerate(rewards):
            discounted_returns += (self.gamma ** t) * r
        return discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `discounted_cumsums`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine
        discounted_cumsums = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum
            discounted_cumsums[t] = running_sum
        return discounted_cumsums
