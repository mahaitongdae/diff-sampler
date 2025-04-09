from rsl_rl.modules import ActorCritic
import torch.nn as nn
import torch
from training.networks import AMEDPredictorWithValue
from torch.nn.functional import silu
from torch.distributions import Normal
import math


class DiffSamplerActorCritic(nn.Module):

    def __init__(
            self,
            hidden_dim = 128,
            output_dim = 1,
            bottleneck_input_dim = 64,
            bottleneck_output_dim = 4,
            noise_channels = 8,
            embedding_type = 'positional',
            dataset_name = None,
            img_resolution = None,
            num_steps = None,
            sampler_tea = None,
            sampler_stu = None,
            M = None,
            guidance_type = None,
            guidance_rate = None,
            schedule_type = None,
            schedule_rho = None,
            afs = False,
            scale_dir = 0,
            scale_time = 0,
            max_order = None,
            predict_x0 = True,
            lower_order_final = True,
            init_noise_std = 1.0,
            **kwargs
    ):
        super(DiffSamplerActorCritic, self).__init__()
        self.shared_network = AMEDPredictorWithValue(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bottleneck_input_dim=bottleneck_input_dim,
            bottleneck_output_dim=bottleneck_output_dim,
            noise_channels=noise_channels,
            embedding_type=embedding_type,
            dataset_name=dataset_name,
            img_resolution=img_resolution,
            num_steps=num_steps,
            sampler_tea=sampler_tea,
            sampler_stu=sampler_stu,
            M=M,
            guidance_type=guidance_type,
            guidance_rate=guidance_rate,
            schedule_type=schedule_type,
            schedule_rho=schedule_rho,
            afs=afs,
            scale_dir=scale_dir,
            scale_time=scale_time,
            max_order=max_order,
            predict_x0=predict_x0,
            lower_order_final=lower_order_final,
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(1, 2))
        self.is_recurrent = False
        self.sigmoid = torch.nn.Sigmoid()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, t_cur, t_next):
        mean, _ = self.shared_network(obs, t_cur, t_next)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
    def act(self, observations, **kwargs):
        x, t_cur, t_next = self.decompose_obs(observations)
        mean, _ = self.shared_network(x, t_cur, t_next)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        actions = self.distribution.sample()
        return self.post_processing_action(actions)

    def post_processing_action(self, actions):
        # r = actions[:, [0]]
        # scale_dir = actions[:, [1]]
        # bounded_r = self.sigmoid(r)
        # bounded_scale_dir = self.sigmoid(scale_dir) / (1 / (2 * self.shared_network.scale_dir)) + (1 - self.shared_network.scale_dir)
        # return torch.cat([bounded_r, bounded_scale_dir], dim=1)
        return torch.tanh(actions)

    def reverse_transform_actions(self, actions, epsilon=1e-6):
        # bounded_scale_dir = actions[:, [1]]
        # bounded_r = actions[:, [0]]
        # bounded_scale_dir = (bounded_scale_dir - (1 - self.shared_network.scale_dir)) / (2 * self.shared_network.scale_dir)
        # squashed_actions = torch.cat([bounded_r, bounded_scale_dir], dim=1)
        # true_actions = -1. * torch.log(torch.reciprocal(squashed_actions) - 1.)
        return 0.5 * (torch.log1p(actions + epsilon) - torch.log1p(-actions + epsilon))  # arctanh(a)


    def get_actions_log_prob(self, actions):
        # raise NotImplementedError
        # r = actions[:, [0]]
        # scale_dir = actions[:, [1]]
        true_actions = self.reverse_transform_actions(actions)
        logp = self.distribution.log_prob(true_actions) - (2 * (math.log(2.) - true_actions - torch.nn.functional.softplus(-2 * actions)))
        return logp.sum(dim=-1)

    def act_inference(self, observations):
        x, t_cur, t_next = self.decompose_obs(observations)
        actions_mean, _ = self.shared_network(x, t_cur, t_next)
        return torch.tanh(actions_mean)

    def evaluate(self, critic_observations, **kwargs):
        x, t_cur, t_next = self.decompose_obs(critic_observations)
        _, value = self.shared_network(x, t_cur, t_next)
        return value

    def decompose_obs(self, obs):
        unet_bottleneck = obs[:, :-2]
        t_cur = obs[0, -2]
        t_next = obs[0, -1]
        return unet_bottleneck, t_cur, t_next

