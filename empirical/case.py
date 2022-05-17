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
from utils.configue import Args
from transformers.trainer_utils import get_last_checkpoint
import utils.tool
from utils.configue import Configure
from utils.cascade_dataset import CascadeDataset
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
import random
from tqdm import tqdm

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
    targs = Args()
    targs.input_max_length = 1024
    targs.generation_max_length = 128
    # We wrap the "string" seq2seq data into "tokenized tensor".
    test_dataset = CascadeDataset(args, targs, model_tokenizer,
                                  seq2seq_test_dataset) if seq2seq_test_dataset else None
    print(seq2seq_test_dataset[0]['seq_out'])
    random.seed(12321)
    slice = random.sample(range(len(seq2seq_test_dataset)), 100)
    os.makedirs(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/case_study',exist_ok=True)
    for order in tqdm(slice):
        with open(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/case_study/{order}.txt','w') as f:
            f.write('input:'+model_tokenizer.batch_decode(test_dataset[order]['input_ids'].unsqueeze(0), skip_special_tokens=True)[0]+'\n')
            f.write('gt:'+seq2seq_test_dataset[order]['seq_out']+'\n')
    for cas_step in range(3):
        if os.path.exists(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_lang.pk'):
            predictions = torch.load(
                f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_lang.pk')
        else:
            predictions=torch.load(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_generation.pk')
            predictions=model_tokenizer.batch_decode(predictions, skip_special_tokens=True)
            torch.save(predictions,f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_lang.pk')
        print('cas_step:',cas_step)
        for order in tqdm(slice):
            with open(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/case_study/{order}.txt', 'a') as f:
                f.write(f'{cas_step}:' + predictions[order] + '\n')



if __name__ == "__main__":
    main()
