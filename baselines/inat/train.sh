#!/usr/bin/env bash
export ALGO=inat
export output_dir=output/${ALGO}_${TASK}
rm -rf ${output_dir}
fairseq-train \
    data/levenshtein/${TASK}/bin/ \
    --save-dir ${output_dir} \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --arch iterative_nonautoregressive_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 3000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --train-step 3 \
    --dae-ratio 0.5 \
    --stochastic-approx \
    --apply-bert-init \
    --source-lang src  \
    --target-lang tgt  \
    --truncate-source \
    --log-format 'simple' --log-interval 300 \
    --fixed-validation-seed 7 \
    --max-source-positions 1026 \
    --max-target-positions 130 \
    --max-tokens 8000 \
    --save-interval-updates 50000 \
    --max-update 100000 \
    --save-interval 5000 \
    --validate-interval	5000 \
    | grep -v libnat_cuda

python -m baselines.inat.generate

export WANDB_API_KEY="<your wandb key>"
export WANDB_PROJECT=LUCAS
export WANDB_ENTITY="your wandb entity"
export WANDB_RUN_GROUP=${ALGO}

python -m baselines.levenshtein.evaluate