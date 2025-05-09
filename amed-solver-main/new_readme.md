
## Installation
```shell
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```


## generate fid

```.bash
# FID evaluation
python fid.py calc --images=/home/naliseas-workstation/Documents/haitong/diff-sampler/amed-solver-main/rl_samples/afhqv2/rl_nfe6 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz
```

## Sample

```.bash
torchrun --standalone --nproc_per_node=2 --master_port=11111 sample_rl.py --exp_dir='exps/2025-04-22/12-08-17-afhqv2-10' --seeds="0-49999"
```