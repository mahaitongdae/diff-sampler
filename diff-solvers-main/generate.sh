SOLVER_FLAGS="--solver=euler --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=1"
python sample.py --dataset_name="afhqv2" --batch=128 --seeds="0-49999" $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS