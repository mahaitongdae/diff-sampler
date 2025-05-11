torchrun --standalone --nproc_per_node=2 --master_port=11111 sample_rl.py --exp_dir='exps/2025-05-09/12-31-43-afhqv2-10' --seeds="0-49999"

CUDA_VISIBLE_DEVICES=1 python sample_rl.py --exp_dir='exps/2025-05-09/21-37-19-afhqv2-10' --seeds="0-49999"