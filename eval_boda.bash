#!/bin/bash

#SBATCH --job-name=mistral-eval # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8 



echo "starting Evaluation......................."
###################### RUN LLM Eval ######################
# FULL_MODEL_NAME='cognitivecomputations/dolphin-2.8-mistral-7b-v02'
FULL_MODEL_NAME="mistralai/Mistral-7B-v0.1"
# FULL_MODEL_NAME="mistralai/Mixtral-8x7B-v0.1"
# FULL_MODEL_NAME="Meta-Llama-3-8B"
# FULL_MODEL_NAME="Meta-Llama-3-8B-Instruct"


TEST_DATASET="boda/kaneko_data"
SHOTS=0
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR="output"
EXPLICIT=1

# CHECKPOINT_PATH="/l/users/abdelrahman.sadallah/UWFE-Mixtral/cognitivecomputations/dolphin-2.8-mistral-7b-v02/best"
# CHECKPOINT_PATH="/l/users/abdelrahman.sadallah/UWFE-Mixtral/mistralai/Mistral-7B-v0.1/best/"
CHECKPOINT_PATH='/l/users/abdelrahman.sadallah/UWFE-mistral-explicit-errors/mistralai/Mistral-7B-v0.1/best'
# CHECKPOINT_PATH='/l/users/abdelrahman.sadallah/UWFE-mixtral-explicit-errors/mistralai/Mixtral-8x7B-v0.1/checkpoint-18500/'
echo " $SHOTS shot,    -    $EXPLICIT  explicit errors,  -    $FULL_MODEL_NAME"


# python evaluate_llm.py \
torchrun --nproc_per_node 1 --master_port=7875 evaluate_llm.py \
--model_name=$FULL_MODEL_NAME \
--per_device_train_batch_size=4 \
--num_shots $SHOTS \
--explicit_errors $EXPLICIT \
--save_file=$SAVE_DIR \
--dataset=$TEST_DATASET \
--checkpoint_path=$CHECKPOINT_PATH \
--output_dir=$SAVEDIR \
--prompt_key="prompt" 



echo " ending " 