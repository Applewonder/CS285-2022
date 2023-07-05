from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return self.log_alpha.exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        obs = ptu.from_numpy(obs)
        dist = self.forward(obs)
        if sample:
            action = dist.sample()
        else:
            action = dist.mean()
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        action_distribution = super().forward(observation)
        mean = action_distribution.mean()
        std = action_distribution.std()
        log_std = torch.log(std)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        action_distribution = SquashedNormal(mean, std.exp(log_std))
        return action_distribution


    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        dist = self.forward(ptu.from_numpy(obs))
        action = self.get_action(obs)
        log_prob = dist.log_prob(action)

        q1, q2 = critic(obs, action)
        min_q = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - min_q).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss, alpha_loss, self.alpha()