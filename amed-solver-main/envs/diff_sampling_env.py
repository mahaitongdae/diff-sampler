import os

import sys
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from solver_utils import *
from solvers_amed import get_denoised, init_hook, get_amed_prediction
from sample import StackedRandomGenerator

from rsl_rl.env import VecEnv
import torch

def amed_sampler(
        net,
        latents,
        class_labels=None,
        condition=None,
        unconditional_condition=None,
        num_steps=None,
        sigma_min=0.002,
        sigma_max=80,
        schedule_type='polynomial',
        schedule_rho=7,
        afs=False,
        denoise_to_zero=False,
        return_inters=False,
        AMED_predictor=None,
        step_idx=None,
        train=False,
        **kwargs
):
    """
    AMED-Solver (https://arxiv.org/abs/2312.00094).

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings.
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        AMED_predictor: A predictor network.
        step_idx: A `int`. An index to specify the sampling step for training.
        train: A `bool`. In the training loop?
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """
    assert AMED_predictor is not None

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type,
                           schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)

        # Euler step.
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur ** 2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition,
                                    unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs,
                                                       batch_size=latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition,
                                unconditional_condition=unconditional_condition)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition,
                              unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next

class DiffSamplingEnv(object):

    def __init__(self,
                 net,
                 # batch_seeds,
                 dataset_name,
                 batch_size,
                 num_steps,
                 schedule_type,
                 schedule_rho,
                 outdir=None,
                 device='cuda',
                 reward_scale=100,
                 **kwargs):
        # Pick latents and labels.
        self.device = torch.device(device)
        self.batch_size = self.num_envs = batch_size
        self.num_actions = 2

        assert dataset_name in ['cifar10']
        self.dataset_name = dataset_name
        # if dataset_name == 'cifar10':
        self.net = net
        self.t_steps = get_schedule(num_steps,
                                    self.net.sigma_min,
                                    self.net.sigma_max,
                                    schedule_type=schedule_type,
                                    schedule_rho=schedule_rho,
                                    net=net,
                                    device=self.device)
        """
        Besides, for AMED-Solver on CIFAR10 32×32 [21], 
        FFHQ 64×64 [17] and ImageNet 64×64 [37],
         we use uniform time schedule which is widely used 
         in papers with a DDPM backbone [15].
        
        in the paper.
        """
        self.solver_kwargs = kwargs
        self.reward_scale = reward_scale
        self.reset(batch_seeds = range(self.batch_size))

    def reset(self, batch_seeds = None):
        if batch_seeds is None:
            batch_seeds = range(self.batch_size)
        self.rnd = StackedRandomGenerator(self.device, batch_seeds)
        self.latents = self.rnd.randn([self.batch_size,
                                       self.net.img_channels,
                                       self.net.img_resolution,
                                       self.net.img_resolution],
                            device=self.device)
        self.x = self.latents.clone() * self.t_steps[0]
        class_labels = c = uc = None
        self.current_step = 0
        self.class_labels, self.c, self.uc = self.__reset_class_labels(batch_seeds)
        self.traj_mean = None
        self.sum_of_square = None

        # Get UNet enc out as observations.
        unet_enc_out, hook = init_hook(self.net, self.class_labels)
        _ = get_denoised(self.net, self.x, self.t_steps[0],
                                class_labels=self.class_labels,
                                condition=self.c,
                                unconditional_condition=self.uc)
        hook.remove()
        self.observation = torch.mean(unet_enc_out[-1], dim=1)
        return self.latents, {}

    def get_observations(self):

        if self.observation is not None and self.x is not None:
            flatten_obs = self.observation.reshape(self.batch_size, -1)
            t_cur = self.t_steps[self.current_step] * torch.ones([self.batch_size, 1]).to(self.device)
            t_next = self.t_steps[self.current_step + 1] * torch.ones([self.batch_size, 1]).to(self.device)
            whole_obs = torch.cat([flatten_obs, t_cur, t_next], dim=1)
            return (whole_obs,
                    {'observations':{'critic': whole_obs}})
        else:
            raise ValueError('Please first call reset')
        
    def preprocess_action(self, action):
        """
        preprocess actions. Usually the actions are between [-1, 1],
        while we reshape it to 
        r: [0, 1]
        scale_dir: [0.8, 1.2]
        """
        raw_r, raw_scale_dir = action[:, 0], action[:, 1]
        r = 0.5 * (raw_r + 1.)
        scale_dir = 1. + 0.2 * raw_scale_dir
        return r, scale_dir
        


    def step(self, action):
        r, scale_dir = self.preprocess_action(action)
        r = r.reshape(-1, 1, 1, 1)
        scale_dir = scale_dir.reshape(-1, 1, 1, 1)
        scale_time = torch.ones_like(scale_dir, device=self.device)
        t_cur = self.t_steps[self.current_step]
        t_next = self.t_steps[self.current_step + 1]
        x_cur = self.x
        unet_enc_out, hook = init_hook(self.net, self.class_labels)

        # By default we set use_afs = False.
        denoised = get_denoised(self.net, x_cur, t_cur,
                                class_labels=self.class_labels,
                                condition=self.c,
                                unconditional_condition=self.uc)

        d_cur = (x_cur - denoised) / t_cur
        hook.remove()
        self.observation = torch.mean(unet_enc_out[-1], dim=1)
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)

        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(self.net, x_next, scale_time * t_mid,
                                class_labels=self.class_labels,
                                condition=self.c,
                                unconditional_condition=self.uc)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid
        self.x = x_next

        rewards = self.get_rewards(denoised)
        # rewards = torch.linalg.norm(d_mid, dim)
        dones = self.get_dones()
        obs = self.get_observations()[0]
        self.current_step += 1

        return obs, rewards, dones, {"observations": {}}

    def step_euler(self):
        t_cur = self.t_steps[self.current_step]
        t_next = self.t_steps[self.current_step + 1]
        x_cur = self.x
        unet_enc_out, hook = init_hook(self.net, self.class_labels)

        # By default we set use_afs = False.

        denoised = get_denoised(self.net, x_cur, t_cur,
                                class_labels=self.class_labels,
                                condition=self.c,
                                unconditional_condition=self.uc)

        d_cur = (x_cur - denoised) / t_cur
        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)

        # t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_next - t_cur) * d_cur
        self.x = x_next

        rewards = self.get_rewards(denoised)
        dones = self.get_dones()
        self.current_step += 1

        return (unet_enc_out, t_cur, t_next), rewards, dones, {}


    def __reset_class_labels(self, batch_seeds):
        solver_kwargs = self.solver_kwargs
        class_labels = c = uc = None
        if self.net.label_dim:
            if solver_kwargs['model_source'] == 'adm':  # ADM models
                class_labels = self.rnd.randint(self.net.label_dim, size=(self.batch_size,), device=self.device)
            elif solver_kwargs['model_source'] == 'ldm' and solver_kwargs['dataset_name'] == 'ms_coco':
                if solver_kwargs['prompt'] is None:
                    # prompts = sample_captions[batch_seeds[0]:batch_seeds[-1] + 1]
                    raise NotImplementedError('LDM and Stable Diffusion codebase are not yet implemented.')
                else:
                    prompts = [solver_kwargs['prompt'] for i in range(self.batch_size)]
                if solver_kwargs['guidance_rate'] != 1.0:
                    uc = self.net.model.get_learned_conditioning(self.batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = self.net.model.get_learned_conditioning(prompts)
            else:
                class_labels = torch.eye(self.net.label_dim, device=self.device)[
                    self.rnd.randint(self.net.label_dim, size=[self.batch_size], device=self.device)]
        return class_labels, c, uc

    def get_rewards_variance(self, denoised):
        n = self.current_step + 1
        if self.traj_mean is None:
            self.traj_mean = denoised.clone()
            self.sum_of_square = denoised.clone() ** 2
            self.mean_flatten_var = torch.zeros([self.batch_size,]).to(self.device)
        else:
            self.traj_mean = self.traj_mean + 1 / n * (denoised - self.traj_mean)
            self.sum_of_square = self.sum_of_square + denoised ** 2
        traj_var = self.sum_of_square / n - self.traj_mean ** 2
        new_mean_flatten_var = torch.mean(torch.reshape(traj_var, [self.batch_size, -1]), dim=1)
        reward = new_mean_flatten_var - self.mean_flatten_var
        self.mean_flatten_var = new_mean_flatten_var
        return self.reward_scale * reward
    
    def get_current_mean_flatten_var(self):
        return self.mean_flatten_var

    def get_dones(self):
        if self.current_step != len(self.t_steps) - 2:
            return torch.zeros([self.batch_size], device=self.device)
        else:
            return torch.ones([self.batch_size], device=self.device)

    def save_images(self, img = None, grid=True, outdir=None):
        from torchvision.utils import make_grid, save_image
        images = self.x if img is None else img
        if outdir is None:
            if grid:
                outdir = os.path.join(f"./samples/grids/{self.dataset_name}", f"env_rollout_nfe{len(self.t_steps) - 1}")
            else:
                outdir = os.path.join(f"./samples/{self.dataset_name}", f"env_rollout_nfe{len(self.t_steps) - 1}")
        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))

    def sample_via_sample_fn(self,):
        from solvers_amed import euler_sampler
        img = euler_sampler(self.net, self.latents, self.class_labels, num_steps=1000, schedule_type='time_uniform',
                        schedule_rho=1.0)
        return img


if __name__ == '__main__':

    import dnnlib
    from torch_utils.download_util import check_file_by_key
    import pickle
    import torch
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    model_path, classifier_path = check_file_by_key('cifar10', subsubdir='../src')
    with dnnlib.util.open_url(model_path) as f:
        net = pickle.load(f)['ema'].to(torch.device('cuda'))
    net.sigma_min = 0.002
    net.sigma_max = 80.0
    batch_size = 8
    num_steps = 1000
    env = DiffSamplingEnv(
        device=torch.device('cuda'),
        batch_seeds=1,
        dataset_name='cifar10',
        batch_size=batch_size,
        num_steps=num_steps,
        schedule_type='time_uniform',
        schedule_rho=1.0,
        net=net,
    )
    env.reset(batch_seeds=torch.randint(low=0, high=1000, size=(batch_size,)))
    # img = env.sample_via_sample_fn()
    done = False
    rs = []
    for i in tqdm(range(num_steps - 1)):
        enc_out, r, done, _ = env.step_euler()
        rs.append(r.cpu().numpy())
    print(env.current_step, done)
    plt.plot(rs)
    # plt.show()
    env.save_images()



