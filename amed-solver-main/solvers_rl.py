import torch
from solver_utils import *
from solvers_amed import get_denoised # , get_amed_prediction
from agents.config import get_args, class_to_dict
from envs.diff_sampling_env import DiffSamplingEnv
from envs.diff_sampling_env_step import DiffSamplingEnvStep
from tqdm import tqdm

def rl_sampler(
    net, 
    latents,
    cfg,
    device,
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
    rl_runner=None, 
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
    assert rl_runner is not None
    
    env = DiffSamplingEnvStep(net=net,
                        dataset_name=cfg.dataset_name,
                        device=device,
                        num_steps=num_steps,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        schedule_type=schedule_type,
                        schedule_rho=schedule_rho,
                        batch_size=latents.shape[0],
                        reward_type = 'norm'
                        # **class_to_dict(cfg.env)
                        )
    
    enc_out, _ = env.force_reset(latent=latents, c=condition, uc=unconditional_condition, class_labels=class_labels)
    policy = rl_runner.alg.actor_critic.act_inference
    
    for i in range(cfg.env.num_steps - 1):
        with torch.no_grad():
            # act = policy(enc_out)
            act = torch.randn([enc_out.shape[0], 2], device=device)
        enc_out, r, done, _ = env.step(act)
        
        if torch.all(done):
            break
        
    return env.x
    
    # # Time step discretization.
    # t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # # Main sampling loop.
    # x_next = latents * t_steps[0]
    # inters = [x_next.unsqueeze(0)]
    # for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
    #     x_cur = x_next
    #     unet_enc_out, hook = init_hook(net, class_labels)
        
    #     # Euler step.
    #     use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
    #     if use_afs:
    #         d_cur = x_cur / ((1 + t_cur**2).sqrt())
    #     else:
    #         denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
    #         d_cur = (x_cur - denoised) / t_cur

    #     hook.remove()
    #     t_cur = t_cur.reshape(-1, 1, 1, 1)
    #     t_next = t_next.reshape(-1, 1, 1, 1)
    #     r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
    #     t_mid = (t_next ** r) * (t_cur ** (1 - r))
    #     x_next = x_cur + (t_mid - t_cur) * d_cur

    #     # Apply 2nd order correction.
    #     denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
    #     d_mid = (x_next - denoised) / t_mid
    #     x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid
    
    #     if return_inters:
    #         inters.append(x_next.unsqueeze(0))
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next