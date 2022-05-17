import setproctitle
setproctitle.setproctitle('SKG')
import logging
import os
os.sys.path.insert(0,'')
import copy
import torch
import collections
if int(torch.__version__.split('.')[1]) >= 8:
    torch._six.container_abcs=collections.abc
import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
import utils.tool
import numpy as np
from utils.configue import Configure
from utils.cascade_dataset import CascadeDataset
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from collections import Counter
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
    set_seed(2)
    args = Configure.Get(f"Salesforce/T5_base_prefix_{os.environ['TASK']}.cfg")
    args.max_cascade_steps=3

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
    model_tokenizer = AutoTokenizer.from_pretrained("t5-base",use_fast=False)
    if args.special_tokens:
        model_tokenizer.add_tokens([v for k, v in args.special_tokens])

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")
    print(seq2seq_test_dataset[0]['seq_out'])
    pbs=[]
    pred=[]
    for cas_step in range(3):
        pb=torch.load(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_generation.pk.pb')
        pbs.append(pb)
        if os.path.exists(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_lang.pk'):
            predictions = torch.load(
                f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_lang.pk')
        else:
            predictions=torch.load(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_generation.pk')
            predictions=model_tokenizer.batch_decode(predictions, skip_special_tokens=True)
            torch.save(predictions,f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_lang.pk')
        pred.append(predictions)
    rank=np.float32(pbs).argmin(0)
    predictions=[pred[r][i] for i,r in enumerate(rank)]
    c=Counter(rank)
    print(c.get(0,0)/len(rank),c.get(1,0)/len(rank),c.get(2,0)/len(rank))
    print(evaluator.evaluate(predictions, seq2seq_test_dataset, "test"))


if __name__ == "__main__":
    main()
# webqsp
# 0.003050640634533252 0.009151921903599756 0.987797437461867
# {'META_TUNING/webqsp.cfg/F1': 0.7481163086816297, 'avr': 0.7481163086816297}

# mtop
# 0.04377564979480164 0.06999544003647971 0.8862289101687186
# {'META_TUNING/mtop.cfg/exact_match': 0.815093479252166, 'META_TUNING/mtop.cfg/template_accuracy': 0.8531691746466028, 'avr': 0.8341313269493844}

# kvret
# 0.18811881188118812 0.14727722772277227 0.6646039603960396
# {'META_TUNING/kvret.cfg/bleu': 0.18694765265471117, 'META_TUNING/kvret.cfg/all_micro': 0.694300518134715, 'META_TUNING/kvret.cfg/all_macro': 0.6688297540520894, 'META_TUNING/kvret.cfg/schedule_micro': 0.7768595041322314, 'META_TUNING/kvret.cfg/schedule_macro': 0.7543214232203496, 'META_TUNING/kvret.cfg/navigate_micro': 0.6442641946697566, 'META_TUNING/kvret.cfg/navigate_macro': 0.6142025696243785, 'META_TUNING/kvret.cfg/weather_micro': 0.6673387096774194, 'META_TUNING/kvret.cfg/weather_macro': 0.6767983647560384, 'avr': 0.6315402989912989}