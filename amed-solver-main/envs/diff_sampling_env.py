from solver_utils import *
from solvers_amed import get_denoised, init_hook, get_amed_prediction
from sample import StackedRandomGenerator
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

    def __init__(self, device,
                 batch_seeds,
                 batch_size,
                 img_channels,
                 img_resolution,
                 time_schedule,
                 num_steps,
                 sigma_min,
                 sigma_max,
                 schedule_type,
                 schedule_rho,
                 net,
                 **solver_kwargs):
        # Pick latents and labels.
        self.device = device
        self.batch_size = batch_size
        self.img_channels = img_channels
        self.image_resolution = img_resolution
        self.time_schedule = time_schedule
        self.net = net
        self.t_steps = get_schedule(num_steps,
                                    sigma_min,
                                    sigma_max,
                                    schedule_type=schedule_type,
                                    schedule_rho=schedule_rho,
                                    net=net,
                                    device=device)
        """
        Besides, for AMED-Solver on CIFAR10 32×32 [21], 
        FFHQ 64×64 [17] and ImageNet 64×64 [37],
         we use uniform time schedule which is widely used 
         in papers with a DDPM backbone [15].
        
        in the paper.
        """
        self.solver_kwargs = solver_kwargs
        pass

    def reset(self, batch_seeds = None):
        self.rnd = StackedRandomGenerator(self.device, batch_seeds)
        self.latents = self.rnd.randn([self.batch_size, self.img_channels, self.image_resolution, self.image_resolution],
                            device=self.device)
        self.x = self.latents.clone()
        class_labels = c = uc = None
        self.current_step = 0
        self.class_labels, self.c, self.uc = self.__reset_class_labels(batch_seeds)
        return self.latents, {}


    def step(self, action):
        r, scale_dir, scale_time = action
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

        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(self.net, x_next, scale_time * t_mid,
                                class_labels=self.class_labels,
                                condition=self.c,
                                unconditional_condition=self.uc)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid

        rewards = self.get_rewards()
        dones = self.get_dones()

        return x_next, rewards, dones, {}



    def __reset_class_labels(self, batch_seeds):
        solver_kwargs = self.solver_kwargs
        class_labels = c = uc = None
        if self.net.label_dim:
            if solver_kwargs['model_source'] == 'adm':  # ADM models
                class_labels = self.rndrnd.randint(self.net.label_dim, size=(self.batch_size,), device=self.device)
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

    def get_rewards(self):
        pass

    def get_dones(self):
        pass