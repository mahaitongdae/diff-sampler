import os
import re
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from agents.diff_sample_runner import DiffSamplerOnPolicyRunner
from agents.config import get_args, class_to_dict
from envs.diff_sampling_env import DiffSamplingEnv
from sample import create_model
import hydra
from omegaconf import OmegaConf, DictConfig
import logging
from hydra.core.hydra_config import HydraConfig

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------

# @click.command()
#
# # General options.
# @click.option('--dataset_name',     help='Dataset name', metavar='STR',                                type=click.Choice(['cifar10', 'ffhq', 'afhqv2', 'imagenet64', 'lsun_bedroom', 'lsun_cat', 'imagenet256', 'ms_coco', 'lsun_bedroom_ldm']), required=True,
#                 default='cifar10')
# @click.option('--outdir',           help='Where to save the results', metavar='DIR',                   type=str, default='./exps')
# @click.option('--total_kimg',       help='Number of images (k) for training', metavar='INT',           type=int, default=10)
# @click.option('--prompt_path',      help='Path to MS-COCO_val2014_30k_captions.csv', metavar='DIR',    type=str)
#
# # Options for solvers
# @click.option('--num_steps',        help='Number of time steps for training', metavar='INT',           type=click.IntRange(min=1), default=4, show_default=True)
# @click.option('--sampler_stu',      help='Student solver', metavar='STR',                              type=click.Choice(['amed', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='amed', show_default=True)
# @click.option('--sampler_tea',      help='Teacher solver', metavar='STR',                              type=click.Choice(['heun', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='heun', show_default=True)
# @click.option('--M',                help='Steps to insert between two adjacent steps', metavar='INT',  type=click.IntRange(min=1), default=1, show_default=True)
# @click.option('--guidance_type',    help='Guidance type',                                              type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
# @click.option('--guidance_rate',    help='Guidance rate', metavar='FLOAT',                             type=float, default=0.)
# @click.option('--schedule_type',    help='Time discretization schedule', metavar='STR',                type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
# @click.option('--schedule_rho',     help='Time step exponent', metavar='FLOAT',                        type=click.FloatRange(min=0), default=7, show_default=True)
# @click.option('--afs',              help='Whether to use afs', metavar='BOOL',                         type=bool, default=True, show_default=True)
# @click.option('--scale_dir',        help='Scale the gradient by [1-scale_dir, 1+scale_dir]', metavar='FLOAT',     type=click.FloatRange(min=0), default=0.01, show_default=True)
# @click.option('--scale_time',       help='Scale the gradient by [1-scale_time, 1+scale_time]', metavar='FLOAT',   type=click.FloatRange(min=0), default=0, show_default=True)
# # Additional options for multi-step solvers, 1<=max_order<=4 for iPNDM, 1<=max_order<=3 for DPM-Solver++
# @click.option('--max_order',        help='max order for solvers', metavar='INT',                       type=click.IntRange(min=1), default=3)
# # Additional options for DPM-Solver++
# @click.option('--predict_x0',       help='Whether to use data prediction mode', metavar='BOOL',        type=bool, default=True)
# @click.option('--lower_order_final',help='Lower the order at final stages', metavar='BOOL',            type=bool, default=True)
#
# # Hyperparameters.
# @click.option('--batch',            help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
# @click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
# @click.option('--lr',               help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=5e-3, show_default=True)
#
# # Performance-related.
# @click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
#
# # I/O-related.
# @click.option('--desc',             help='String to include in result dir name', metavar='STR',        type=str)
# @click.option('--nosubdir',         help='Do not create a subdirectory for results',                   is_flag=True)
# @click.option('--tick',             help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=10, show_default=True)
# @click.option('--snap',             help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
# @click.option('--dump',             help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
# @click.option('--seed',             help='Random seed  [default: random]', metavar='INT',              type=int)
# @click.option('-n', '--dry-run',    help='Print training options and exit',                            is_flag=True)

@hydra.main(config_path='conf', config_name='edm_afhq2.yaml')
def main(cfg):
    OmegaConf.resolve(cfg)
    print(cfg)
    # return
    # torch.multiprocessing.set_start_method('spawn')
    # dist.init()

    # c = dnnlib.EasyDict()
    # c.loss_kwargs = dnnlib.EasyDict()
    # c.AMED_kwargs = dnnlib.EasyDict()
    # c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # AMED predictor architecture.
    # c.AMED_kwargs.class_name = 'training.networks.AMED_predictor'
    # c.AMED_kwargs.update(num_steps=opts.num_steps, sampler_stu=opts.sampler_stu, sampler_tea=opts.sampler_tea, \
    #                      M=opts.m, guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate, \
    #                      schedule_rho=opts.schedule_rho, schedule_type=opts.schedule_type, afs=opts.afs, \
    #                      dataset_name=opts.dataset_name, scale_dir=opts.scale_dir, scale_time=opts.scale_time, \
    #                      max_order=opts.max_order, predict_x0=opts.predict_x0, lower_order_final=opts.lower_order_final)
    # c.loss_kwargs.class_name = 'training.loss.AMED_loss'

    # Training options.
    # c.total_kimg = opts.total_kimg      # Train for total_kimg k trajectories
    # c.kimg_per_tick =  1                # total_kimg ticks
    # c.snapshot_ticks = c.total_kimg     # 1 snapshots
    # c.state_dump_ticks = c.total_kimg   # 1 dump
    # c.update(dataset_name=opts.dataset_name, batch_size=opts.batch, batch_gpu=opts.batch_gpu, gpus=dist.get_world_size(), cudnn_benchmark=opts.bench)
    # c.update(guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate, prompt_path=opts.prompt_path)
    
    # Random seed.
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        cfg.seed = int(seed)

    # Description string.
    if cfg.env.schedule_type == 'polynomial':
        schedule_str = 'poly' + str(cfg.env.schedule_rho)
    elif cfg.env.schedule_type == 'logsnr':
        schedule_str = 'logsnr'
    elif cfg.env.schedule_type == 'time_uniform':
        schedule_str = 'uni' + str(cfg.env.schedule_rho)
    elif cfg.env.schedule_type == 'discrete':
        schedule_str = 'discrete'
    else:
        raise ValueError("Got wrong schedule type: {}".format(cfg.schedule_type))
    # Calculate required NFE
    nfe = 2 * (cfg.env.num_steps - 1) - 1 if cfg.afs else 2 * (cfg.env.num_steps - 1)
    nfe = 2 * nfe if cfg.dataset_name == 'ms_coco' else nfe
    if cfg.afs == True:
        desc = f'{cfg.dataset_name:s}-{cfg.env.num_steps}-{cfg.m}-{schedule_str}-afs' # -{nfe}
    else:
        desc = f'{cfg.dataset_name:s}-{cfg.env.num_steps}-{cfg.m}-{schedule_str}' # -{nfe}
    if cfg.desc != "None":
        desc += f'{cfg.desc}'

    # Pick output directory.
    # if dist.get_rank() != 0:
    #     # cfg.run_dir = None
    #     pass
    # elif cfg.nosubdir:
    #     cfg.run_dir = cfg.outdir
    # else:
    prev_run_dirs = []
    if os.path.isdir(cfg.outdir):
        prev_run_dirs = [x for x in os.listdir(cfg.outdir) if os.path.isdir(os.path.join(cfg.outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join('./', f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)

    model_dir = os.path.join(hydra.utils.get_original_cwd(),
                                              'src')
    # if dist.get_rank() == 0:
        # Print options.
    print()
    print('Training options:')
    print(cfg)
    print()
    print(f'Cwd:                     {os.getcwd()}')
    print(f'Output directory:        {os.path.abspath(run_dir)}')
    print(f'Model directory:         {model_dir}')
    print(f'Number of GPUs:          {dist.get_world_size()}')
    print(f'Batch size:              {cfg.env.batch_size}')
    print()

    # Dry run?
    if cfg.dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    # print('Creating output directory...')
    # if dist.get_rank() == 0:
    #     os.makedirs(c.run_dir, exist_ok=True)
    #     with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
    #         json.dump(c, f, indent=2)
    #     dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.

    # Load pre-trained denoising network.
    net = create_model(dataset_name=cfg.dataset_name, device=torch.device(cfg.device),
                       subsubdir=model_dir)[0]
    env = DiffSamplingEnv(net=net,
                          dataset_name=cfg.dataset_name,
                          device=cfg.device,
                          **class_to_dict(cfg.env)
                          )
    Runner = eval(cfg.runner_class_name)
    ppo_runner = Runner(env=env,
                        train_cfg=cfg,
                        device=cfg.device,
                        log_dir=run_dir,)
    ppo_runner.learn(num_learning_iterations=cfg.runner.max_iterations)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
