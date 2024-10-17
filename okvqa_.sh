#!/bin/bash

#SBATCH --job-name=ofa_okvqa                   # Name of the process
#SBATCH --cpus-per-task=2                      # Number of CPU cores (2 is reasonable)
#SBATCH --gres=gpu:1                           # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=64G                              # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --output=/gaueko0/users/ietxarri010/GrAL_Irene/log_OFA_prompt.log
#SBATCH --error=/gaueko0/users/ietxarri010/GrAL_Irene/error_OFA_prompt.err

source /gscratch/users/asalaberria009/env/p39-cu115/bin/activate

#export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
srun python erantzunak_berridatzi.py --model_type "llama-8b" --root /gaueko0/users/ietxarri010/GrAL_Irene/okvqa