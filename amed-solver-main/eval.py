from omegaconf import OmegaConf
import torch
from agents.diff_sample_runner import DiffSamplerOnPolicyRunner
from envs.diff_sampling_env import DiffSamplingEnv
from sample import create_model
from agents.config import get_args, class_to_dict
import os
from tqdm import tqdm

import argparse

base = os.path.dirname(__file__)

# # 1. Path to saved experiment
# exp_dir = "exps/2025-04-09/16-32-43-afhqv2"
# exp_dir = f"{base}/{exp_dir}"
#
# hydra_config_path = f"{exp_dir}/.hydra/config.yaml"
# exp_index = '00000'
# model_steps = '49'
# for dir in os.listdir(exp_dir):
#     if dir.startswith(exp_index):
#         model_dir = f"{exp_dir}/{dir}"
# model_path = f"{model_dir}/model_{model_steps}.pt"

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--euler", type=bool, default=False)
    parser.add_argument("--exp_dir", type=str, default='exps/2025-04-22/12-08-17-afhqv2-10')
    parser.add_argument("--exp_index", type=str, default='00000')
    parser.add_argument("--model_steps", type=str, default='49')
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)


    args = parser.parse_args()
    exp_dir = args.exp_dir
    exp_index = args.exp_index
    model_steps = args.model_steps

    # 2. Load config
    hydra_config_path = f"{exp_dir}/.hydra/config.yaml"
    cfg = OmegaConf.load(hydra_config_path)

    exp_dir = f"{base}/{exp_dir}"

    hydra_config_path = f"{exp_dir}/.hydra/config.yaml"
    for dir in os.listdir(exp_dir):
        if dir.startswith(exp_index):
            model_dir = f"{exp_dir}/{dir}"
    model_path = f"{model_dir}/model_{model_steps}.pt"

    # 3. (Optional) Reconstruct the model using the config
    # This step depends on how your model is defined in the config
    # For example:
    net = create_model(dataset_name=cfg.dataset_name, device=torch.device(cfg.device),
                        subsubdir='./src/')[0]
    cfg.env.batch_size = args.batch_size
    env = DiffSamplingEnv(net=net,
                        dataset_name=cfg.dataset_name,
                        device=cfg.device,
                        # batch_size=8,
                        **class_to_dict(cfg.env)
                        )
    Runner = DiffSamplerOnPolicyRunner
    ppo_runner = Runner(env=env,
                        train_cfg=cfg,
                        device=cfg.device,
                        log_dir=exp_dir,)
    ppo_runner.load(model_path,)
    policy = ppo_runner.alg.actor_critic.act_inference

    enc_out, _ = env.reset()  # batch_seeds=torch.randint(low=0, high=1000, size=(batch_size,))
    # img = env.sample_via_sample_fn()
    done = False
    rs = []
    for i in tqdm(range(cfg.env.num_steps - 1)):
        if args.euler:
            act = torch.zeros([env.batch_size, 2], device=cfg.device)
        else:
            with torch.no_grad():
                act = policy(enc_out)
        enc_out, r, done, _ = env.step(act)
        rs.append(r.cpu().numpy())
    print(env.current_step, done)
    # rs = np.array(rs).sum(axis=0)
    # print(rs)
    # img = torch.clamp(env.get_current_image() / 2 + 0.5, 0, 1)
    # # img = draw_indices_on_images(img, rs)
    #
    # env.save_images(img)
    fname = 'grid.png' if args.euler else f'grid_{model_steps}.png'
    env.save_images(outdir=exp_dir, fname=fname)

if __name__ == '__main__':
    eval()


