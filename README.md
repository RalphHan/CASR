This is the implement of the paper, **CASR: Generating Complicated Sequences with
Autoregressive Self-Boost Refinement**.

The overview logic is in [utils.cascade_trainer.CascadeSeq2SeqTrainer.train_all](utils/cascade_trainer.py)
# Quick Start
build and run the docker image with [Dockerfile_sing](Dockerfile_sing) for fine-tuning and [Dockerfile_ada](Dockerfile_ada) for adapter-tuning 

set the MASTER_PORT and TASK environment variable:
```
export MASTER_PORT=12345
export TASK=webqsp (or mtop, kvret)
```
To train finetuning+sepenc+continue, run: 
```
bash script/ft-t5-con/run_sing.sh
```
To train continue on Sudoku, run: 
```
BART_SIZE=base bash script/trainer_sudoku.sh
```

To train CASR-Large, run:
```
bash script/trainer_large.sh
```

# Baselines 
```
bash baselines/inat/train.sh # INAT
bash baselines/levenshtein/train.sh # Levenshtein
bash baselines/bidirectional/train.sh # Bidirectional
python -m baselines.progressive.prepare_vocab && bash baselines/progressive/train.sh # Progressive
```

# Empirical Studies
Scripts of empirical studies are in this directory [empirical](empirical)

# More Case Studies
More cases are in this directory [cases](cases)