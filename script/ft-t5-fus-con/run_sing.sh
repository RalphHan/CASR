#!/usr/bin/env bash
export EXP_NAME=fusenc
export SERVICE_NAME=sing
export CLUSTER_NAME=msrresrchvc
#name of your cluster, run "amlt target info amlk8s" to select

export GIT_BRANCH=origin/main
#branch of your git repo, a "git clone" will be run remotely.
if [[ ! -z ${DO_SUBMIT} ]];
then
  export SKU=16G4-V100
  export TASK_CHOICE='"spider_with_cell_value","grailqa","webqsp","mtop","fetaqa","kvret","cosql_with_cell_value"'
  #which seeds do you use

  export MAX_TRIALS=`python -c "print(len([$TASK_CHOICE]))"`
  #number of seeds(automatically computed)
  export TASK_CODE=$0
  #the entrance, which your need to realise for pretrain and finetune
  unset DISPLAY
  cd amlt
  amlt run run.yaml ${EXP_NAME}
  #amlt run <config>.yaml <experiment-name>
  exit 0
fi

#WANDB config
export WANDB_API_KEY="<your wandb key>"
export WANDB_PROJECT=CAIR
export WANDB_ENTITY="your wandb entity"
export WANDB_RUN_GROUP=${EXP_NAME}_finetune_continue
export WANDB_RUN_NOTES="running ${EXP_NAME} with task ${TASK}"

STEP_DIVIDE=1

#WANDB config
export HOST_NAME=`hostname`
TRAINER_STATE=output/sepenc_T5_base_finetune_${TASK}_restart/cas_0/trainer_state.json
PREFIX_CKPT=`python -c "import json;f=open(\"${TRAINER_STATE}\");state=json.load(f);print(state[\"best_model_checkpoint\"]);f.close()"`/pytorch_model.bin
rm -rf output/${EXP_NAME}_T5_base_finetune_${TASK}_continue

export MKL_SERVICE_FORCE_INTEL=1
export TOKENIZERS_PARALLELISM=false

#the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}
export COMMIT_HASH=`git log -n1 --format=format:"%H"`
#record the commit hash

declare -A MAX_STEPS
#MAX_STEPS=([spider_with_cell_value]=16500 [grailqa]=17000 [webqsp]=1500 [mtop]=30000 [fetaqa]=11000 [kvret]=4000 [cosql_with_cell_value]=38000)
MAX_STEPS=([spider_with_cell_value]=4000 [grailqa]=4000 [webqsp]=1500 [mtop]=4000 [fetaqa]=4000 [kvret]=4000 [cosql_with_cell_value]=4000)

python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=${MASTER_PORT} \
--use_env \
cascade_train_finetune.py \
--seed 2 \
--cfg Salesforce/T5_base_prefix_${TASK}.cfg \
--run_name T5_base_finetune_${TASK}_continue \
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
--output_dir output/${EXP_NAME}_T5_base_finetune_${TASK}_continue \
--overwrite_output_dir \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--generation_num_beams 4 \
--generation_max_length 128 \
--input_max_length 1024 \
--ddp_find_unused_parameters true \
--max_cascade_steps 3 \
--dataloader_num_workers 1 \
--lucas_method ${EXP_NAME} \
--fp16 \
--fp16_opt_level O2 \
--start_from_first_castep true \
--load_prefix_from ${PREFIX_CKPT}

