#!/bin/bash

#SBATCH --job-name="SAC_agent"
#SBATCH --output="%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="%j.err" # job standard error file (%j replaced by job id)

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node 
#SBATCH --mem=5G   # maximum memory per node
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu    # gpu node(s)

#========================================================

#srun --unbuffered 

python src/main.py \
--initial_cash 100000 \
--buy_cost 0.0001 \
--sell_cost 0.0001 \
--bank_rate 0.5 \
--sac_temperature 1.0 \
--limit_n_stocks 100 \
--lr_Q 0.0003 \
--lr_pi 0.0003 \
--lr_alpha 0.0003 \
--gamma 0.99 \
--tau 0.005 \
--batch_size 256 \
--layer_size 256 \
--n_episodes 1000 \
--seed 0 \
--delay 2 \
--mode train \
--memory_size 1000000 \
--initial_date 2000-01-03 \
--final_date 2021-12-31 \
--gpu_devices 0 1 2 3 \
--grad_clip 2.0 \
--buy_rule most_first \
--agent_type automatic_temperature \
--window 20 \
--use_corr_matrix \

#--checkpoint_directory saved_outputs/2021.07.26.21.49.56 \ 
