#!/usr/bin/env bash
export HOST_NAME=`hostname`
export EXP_NAME=scratch
export TASK=sudoku
#name of your cluster, run "amlt target info amlk8s" to select
STEP_DIVIDE=1
#WANDB config
export WANDB_API_KEY="<your wandb key>"
export WANDB_PROJECT=CAIR
export WANDB_ENTITY="your wandb entity"
export WANDB_RUN_GROUP=${EXP_NAME}_retry
export WANDB_RUN_NOTES="running ${EXP_NAME} with task ${TASK}"

rm -rf output/bart_${BART_SIZE}_${TASK}_retry

python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=1234 \
--use_env \
cascade_train_sudoku.py \
--seed 2 \
--run_name bart_${BART_SIZE}_${TASK} \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 50 \
--evaluation_strategy steps \
--eval_steps 2000 \
--metric_for_best_model loss \
--greater_is_better false \
--save_strategy steps \
--save_steps 2000 \
--load_best_model_at_end \
--gradient_accumulation_steps 4 \
--max_steps 10000 \
--adafactor true \
--learning_rate 2e-5 \
--do_train --do_eval --do_predict --predict_with_generate \
--output_dir output/bart_${BART_SIZE}_${TASK}_retry \
--overwrite_output_dir \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--eval_accumulation_steps 100 \
--generation_num_beams 2 \
--ddp_find_unused_parameters true \
--max_cascade_steps 5 \
--dataloader_num_workers 1 \
--bart_size ${BART_SIZE}
