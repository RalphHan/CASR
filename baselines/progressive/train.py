import setproctitle
setproctitle.setproctitle('SKG')
import logging
import os
import time
import copy
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
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from transformers.trainer_utils import get_last_checkpoint
import utils.tool
from utils.configue import Configure
from .dataset import ProgressiveDataset
from .trainer import ProgressiveSeq2SeqTrainer
from .model import Model
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
    args = Configure.Get(training_args.cfg)
    args.max_cascade_steps=training_args.max_cascade_steps
    if 'checkpoint-???' in args.bert.location:
        args.bert.location = get_last_checkpoint(
            os.path.dirname(args.bert.location.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {args.bert.location.model_name_or_path}")
    training_args.report_to = ['wandb']
    if "wandb" in training_args.report_to and training_args.process_index <= 0:
        import wandb
        wandb.init(name=training_args.run_name,notes=os.environ.get('WANDB_RUN_NOTES', None))
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

    if not args.arg_paths:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                         cache_dir=args.dataset.data_store_path)
        with FileLock(".lock") as lock:
            seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
                raw_datasets_split, cache_root)
    else:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        meta_tuning_data = {}
        for task, arg_path in args.arg_paths:
            task_args = Configure.Get(arg_path)
            task_args.bert = args.bert
            print('task_args.bert.location:', task_args.bert.location)
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
                path=task_args.dataset.loader_path,
                cache_dir=task_args.dataset.data_store_path)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(
                task_args). \
                to_seq2seq(task_raw_datasets_split, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split
        with FileLock(".lock") as lock:
            seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args). \
                to_seq2seq(meta_tuning_data)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    if training_args.lucas_method:
        assert training_args.lucas_method in {'sepenc','fusenc','casdec'}

    if "t5" in training_args.backbone:
        model = Model.from_pretrained(training_args.backbone)
        if training_args.lucas_method=='sepenc':
            model.encoder2=copy.deepcopy(model.encoder)
    else:
        raise NotImplementedError

    model.policy=training_args.lucas_method
    model.cascade_step=0
    model_tokenizer = AutoTokenizer.from_pretrained(training_args.backbone,use_fast=False)
    if args.special_tokens:
        model_tokenizer.add_tokens([v for k, v in args.special_tokens])
        model.resize_token_embeddings(len(model_tokenizer))

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")

    if training_args.debug_mode:
        seq2seq_train_dataset=[seq2seq_train_dataset[i] for i in range(10)] if seq2seq_train_dataset is not None else None
        seq2seq_eval_dataset=[seq2seq_eval_dataset[i] for i in range(10)] if seq2seq_eval_dataset is not None else None
        seq2seq_test_dataset=[seq2seq_test_dataset[i] for i in range(10)] if seq2seq_test_dataset is not None else None
        training_args.logging_steps=1
        training_args.eval_steps=5
        training_args.save_steps=5
        training_args.max_steps=20

    # We wrap the "string" seq2seq data into "tokenized tensor".
    vocab_weight = torch.load(training_args.vocab_weight_path)
    vocab_mask = torch.full((training_args.max_cascade_steps, len(model_tokenizer)), float('-inf'), dtype=torch.float32)
    vocab_order = sorted(vocab_weight.items(), key=lambda x: x[1], reverse=True)
    vocab_order = [x[0] for x in vocab_order]
    block_size = len(vocab_order) // training_args.max_cascade_steps
    for castep in range(training_args.max_cascade_steps - 1):
        vocab_mask[castep, vocab_order[:(castep + 1) * block_size]] = 0.0
        vocab_mask[castep, 1] = 0.0
    vocab_mask[training_args.max_cascade_steps - 1] = 0.0

    model.register_buffer('vocab_mask',vocab_mask)

    train_dataset = ProgressiveDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset,vocab_mask) if seq2seq_train_dataset else None
    eval_dataset = ProgressiveDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset,vocab_mask) if seq2seq_eval_dataset else None
    test_dataset = ProgressiveDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset,vocab_mask) if seq2seq_test_dataset else None


    # Initialize our Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
    trainer = ProgressiveSeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        test_dataset=test_dataset,
        test_examples=seq2seq_test_dataset,
        callbacks=[early_stopping_callback],
    )
    print('Trainer build successfully.')
    if training_args.load_prefix_from:#hkunlp/T5_base_prefix_all_tasks_2upsample2
        state_dict = torch.load(training_args.load_prefix_from, map_location="cpu")
        msg=trainer.model.load_state_dict(state_dict, strict=False)
        if training_args.process_index <= 0:
            print(msg)
        # release memory
        del state_dict

    # Training
    if training_args.do_train:
        trainer.train_all()


if __name__ == "__main__":
    main()
