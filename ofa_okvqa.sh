#!/bin/bash

#SBATCH --job-name=ofa_okvqa                   # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/GrAL_Irene/log_OFA_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/GrAL_Irene/error_OFA_prompt.err

source /gscratch/users/asalaberria009/env/p39-cu115/bin/activate


#export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

srun python mm_okvqa_finetuning.py --model "OFA-Sys/ofa-base" --target_model ofa --location_encoding none \
   --lr 2e-5 --batch_size 4 --max_steps 20000 --accumulate_grad_batches 2 \
   --run_name ofa_base_okvqa_mc2_syn --train --evaluate --source vinvl --dataset mc2 --synonyms \
   --root /gaueko0/users/ietxarri010/GrAL_Irene/okvqa
