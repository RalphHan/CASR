#!/usr/bin/env bash
export HOST_NAME=`hostname`
export EXP_NAME=fusenc
#name of your cluster, run "amlt target info amlk8s" to select
STEP_DIVIDE=1
#WANDB config
export WANDB_API_KEY="<your wandb key>"
export WANDB_PROJECT=LUCAS
export WANDB_ENTITY="your wandb entity"
export WANDB_RUN_GROUP=bidirectional
export WANDB_RUN_NOTES="running bidirectional with task ${TASK}"
rm -rf output/bidirectional_${TASK}

export MKL_SERVICE_FORCE_INTEL=1
export TOKENIZERS_PARALLELISM=false

#the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}
export COMMIT_HASH=`git log -n1 --format=format:"%H"`
#record the commit hash

declare -A MAX_STEPS
#MAX_STEPS=([spider_with_cell_value]=16500 [grailqa]=17000 [webqsp]=1500 [mtop]=30000 [fetaqa]=11000 [kvret]=4000 [cosql_with_cell_value]=38000)
MAX_STEPS=([spider_with_cell_value]=4000 [grailqa]=4000 [webqsp]=1500 [mtop]=4000 [fetaqa]=4000 [kvret]=4000 [cosql_with_cell_value]=4000)

if [[ -z ${DEBUG} ]];
then
  DEBUG=false
fi

python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=1234 \
--use_env \
-m baselines.bidirectional.train \
--seed 2 \
--cfg Salesforce/T5_base_prefix_${TASK}.cfg \
--run_name bidirectional_${TASK} \
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
--gradient_accumulation_steps 16 \
--max_steps `python -c "print((${MAX_STEPS[$TASK]}//${STEP_DIVIDE}+1999)//2000*2000)"` \
--adafactor true \
--learning_rate 2e-5 \
--do_train --do_eval --do_predict --predict_with_generate \
--output_dir output/bidirectional_${TASK} \
--overwrite_output_dir \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--generation_num_beams 4 \
--generation_max_length 128 \
--input_max_length 1024 \
--ddp_find_unused_parameters true \
--max_cascade_steps 2 \
--dataloader_num_workers 1 \
--lucas_method ${EXP_NAME} \
--fp16 \
--fp16_opt_level O2 \
--do_restart \
--debug_mode ${DEBUG}



