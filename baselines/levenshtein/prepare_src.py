import setproctitle
setproctitle.setproctitle('SKG')
from sklearn.feature_extraction.text import TfidfVectorizer


from tqdm import tqdm
import logging
import os
os.sys.path.insert(0,'')
from utils.configue import Args
import copy
import torch
import collections
if int(torch.__version__.split('.')[1]) >= 8:
    torch._six.container_abcs=collections.abc
import datasets
from transformers import (
    set_seed,
)
from transformers import AutoTokenizer
import utils.tool
from utils.configue import Configure
from utils.cascade_dataset import CascadeDataset
import numpy as np
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
    # parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
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

    # evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)

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

    conv_sep = " || "
    os.makedirs('data/levenshtein/%s'%os.environ['TASK'],exist_ok=True)
    for split,dataset in zip(['train','valid','test'],[seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset]):
        src = []
        tgt = []
        for raw_item in tqdm(dataset):
            if raw_item["text_in"]:
                ###################
                # With text input #
                ###################
                if conv_sep in raw_item["text_in"]:
                    ##################
                    # Conversational #
                    ##################
                    # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                    # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                    index = raw_item["text_in"].index(conv_sep)
                    if args.model.knowledge_usage == 'concatenate' or args.model.knowledge_usage is None:
                        # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                        seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                      raw_item["struct_in"],
                                                                                      raw_item["text_in"][
                                                                                      index + len(conv_sep):])
                    elif args.model.knowledge_usage == 'separate':
                        # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                        seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                           raw_item["text_in"][index + len(conv_sep):])
                    else:
                        raise ValueError()
                else:
                    ######################
                    # Non-conversational #
                    ######################
                    if args.model.knowledge_usage == 'concatenate' or args.model.knowledge_usage is None:
                        # seq_in  = "text_in ; structured knowledge: struct_in"
                        seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                    elif args.model.knowledge_usage == 'separate':
                        # seq_in  = "text_in"
                        seq_in = raw_item["text_in"]
                    else:
                        raise ValueError()
            else:
                ######################
                # Without text input #
                ######################
                if args.model.knowledge_usage == 'concatenate':
                    # seq_in  = "structured knowledge: struct_in"
                    seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
                elif args.model.knowledge_usage == 'separate':
                    # seq_in  = ""
                    seq_in = ""
                else:
                    raise ValueError()

            # Concatenate description.
            if args.model.use_description and args.model.concatenate_description:
                seq_in = "{} ; {}".format(raw_item["description"], seq_in)

            src.append(seq_in+'\n')
            tgt.append(raw_item["seq_out"]+'\n')

        with open('data/levenshtein/%s/%s.src'%(os.environ['TASK'],split),'w') as f:
            f.writelines(src)
        with open('data/levenshtein/%s/%s.tgt'%(os.environ['TASK'],split),'w') as f:
            f.writelines(tgt)


if __name__ == "__main__":
    main()
