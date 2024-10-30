import numpy as np
from rob831.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_values = self.critic.q_net(ptu.from_numpy(observation))
        action = np.argmax(ptu.to_numpy(q_values), axis=1)
        return action.squeeze()
