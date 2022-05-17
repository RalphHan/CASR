import setproctitle

setproctitle.setproctitle('SKG')
import logging
import os
import time

import torch
import collections
if int(torch.__version__.split('.')[1]) >= 8:
    torch._six.container_abcs=collections.abc
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
import utils.tool
from utils.cascade_dataset_sudoku import CascadeDatasetSudoku
from utils.cascade_trainer_sudoku import CascadeSeq2SeqTrainerSudoku
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    from filelock import FileLock
    import nltk
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    training_args.report_to = ['wandb']
    if "wandb" in training_args.report_to and training_args.process_index <= 0:
        import wandb
        wandb.init(name=training_args.run_name, notes=os.environ.get('WANDB_RUN_NOTES', None))
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update({'aml_user': os.environ.get("USER", None),
                             'exp_name': os.environ.get("EXP_NAME", None),
                             'commit_hash': os.environ.get("COMMIT_HASH", None),
                             'cluster': os.environ.get("CLUSTER_NAME", None),
                             'git_branch': os.environ.get("GIT_BRANCH", None),
                             'host_name': os.environ.get("HOST_NAME", None),
                             })

    os.makedirs(training_args.output_dir, exist_ok=True)

    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.

    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path='csv',
                                                                     data_files={
                                                                         k: os.path.join(training_args.sudoku_path,
                                                                                         f'sudoku_{k}.csv') for k in
                                                                         ['train', 'eval', 'test']})

    model = utils.tool.get_model('unified.scratch')(training_args.bart_size)

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = [raw_datasets_split[k] for k in ['train', 'eval', 'test']]


    # seq2seq_train_dataset,seq2seq_eval_dataset,seq2seq_test_dataset=[[x[i] for i in range(10)] for x in [seq2seq_train_dataset,seq2seq_eval_dataset,seq2seq_test_dataset]]

    train_dataset = CascadeDatasetSudoku(seq2seq_train_dataset,training_args.do_non_auto)
    eval_dataset = CascadeDatasetSudoku(seq2seq_eval_dataset,training_args.do_non_auto)
    test_dataset = CascadeDatasetSudoku(seq2seq_test_dataset,training_args.do_non_auto)
    # Initialize our Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=200)
    trainer = CascadeSeq2SeqTrainerSudoku(
        args=training_args,
        model=model,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        callbacks=[early_stopping_callback],
    )
    print('Trainer build successfully.')

    if training_args.load_prefix_from:
        state_dict = torch.load(training_args.load_prefix_from, map_location="cpu")
        msg = trainer.model.load_state_dict(state_dict, strict=False)
        if training_args.process_index <= 0:
            print(msg)
        # release memory
        del state_dict

    # Training
    if training_args.do_train:
        trainer.train_all()


if __name__ == "__main__":
    main()
