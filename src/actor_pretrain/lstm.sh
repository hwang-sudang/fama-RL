

# python3 -m actor_pretrain.main

# cd /home1/hwang/trading14/src/actor_pretrain;

python /home1/hwang/trading2/src/pretrain_main.py \
--task price \
--network GRU \
--norm logr \
--country NASDAQ \
--lr_pi 3e-5 \
--batch_size 128 \
--layer_size 4 \
--n_epochs 3000 \
--seed 42 \
--mode train \
--initial_date 2002-09-03 \
--final_date 2022-12-31 \
--gpu_devices 6 \
--sweep 0 \
--window 60 \
--lr_scheduler reduce 