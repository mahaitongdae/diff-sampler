from rsl_rl.modules import ActorCritic
import torch.nn as nn
import torch
from training.networks import AMEDActorCritic
from torch.nn.functional import silu
from torch.distributions import Normal


class DiffSamplerPolicy(object):

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
    ):

        self.actor_critic = AMEDActorCritic(
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
        self.std = nn.Parameter(init_noise_std * torch.ones(1))

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
        mean, scale_dir, _ = self.actor_critic(obs, t_cur, t_next)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
    def act(self, obs, t_cur, t_next):
        self.update_distribution(obs, t_cur, t_next)
        action = self.distribution.sample()
        return torch.sigmoid(action)

    def get_actions_log_prob(self, actions):
        true_actions = -1. * torch.log(torch.reciprocal(actions) - 1.)
        return self.distribution.log_prob(true_actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean, _, _ = self.actor_critic(observations)
        return torch.sigmoid(actions_mean)

    def evaluate(self, critic_observations, **kwargs):
        _, _, value = self.actor_critic(critic_observations)
        return value

