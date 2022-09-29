#!/usr/bin/env bash
export EXP_NAME=sepenc

export SERVICE_NAME=sing
export CLUSTER_NAME=msrresrchvc
#name of your cluster, run "amlt target info amlk8s" to select
export IMAGE_NAME=ralphhan/lucas:sing-0.1

export GIT_BRANCH=origin/main
#branch of your git repo, a "git clone" will be run remotely.
if [[ ! -z ${DO_SUBMIT} ]];
then
  export SKU=16G4-V100
  export TASK_CHOICE='"webqsp","mtop","kvret"'
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

export HOST_NAME=`hostname`
#name of your cluster, run "amlt target info amlk8s" to select
STEP_DIVIDE=1
#WANDB config
export WANDB_API_KEY="<your wandb key>"
export WANDB_PROJECT=LUCAS
export WANDB_ENTITY="your wandb entity"
export WANDB_RUN_GROUP=T5_base_scratchdec
export WANDB_RUN_NOTES="running T5_base_scratchdec with task ${TASK}"
rm -rf output/T5_base_scratchdec_${TASK}

export MKL_SERVICE_FORCE_INTEL=1
export TOKENIZERS_PARALLELISM=false

#the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}
export COMMIT_HASH=`git log -n1 --format=format:"%H"`
#record the commit hash

declare -A MAX_STEPS
MAX_STEPS=([webqsp]=2000 [mtop]=4000 [kvret]=4000)
declare -A SAVE_STEPS
SAVE_STEPS=([webqsp]=2000 [mtop]=2000 [kvret]=2000)

if [[ -z ${DEBUG} ]];
then
  DEBUG=false
fi

python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=1234 \
--use_env \
 cascade_train_finetune.py \
--seed 2 \
--cfg Salesforce/T5_base_prefix_${TASK}.cfg \
--run_name T5_base_scratchdec_${TASK} \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 50 \
--evaluation_strategy steps \
--eval_steps ${SAVE_STEPS[$TASK]} \
--metric_for_best_model loss \
--greater_is_better false \
--save_strategy steps \
--save_steps ${SAVE_STEPS[$TASK]} \
--load_best_model_at_end \
--gradient_accumulation_steps 16 \
--max_steps ${MAX_STEPS[$TASK]} \
--adafactor true \
--learning_rate 2e-5 \
--do_train --do_eval --do_predict --predict_with_generate \
--output_dir output/T5_base_scratchdec_${TASK} \
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
--scratch_decoder true \
--debug_mode ${DEBUG}



