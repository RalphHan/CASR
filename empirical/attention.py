import setproctitle
setproctitle.setproctitle('SKG')
from tqdm import tqdm
import logging
import os
os.sys.path.insert(0,'')
from utils.configue import Args
import time
import copy
import torch
import collections
if int(torch.__version__.split('.')[1]) >= 8:
    torch._six.container_abcs=collections.abc
import datasets
import transformers
from torch.nn import CrossEntropyLoss
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from transformers.trainer_utils import get_last_checkpoint
import utils.tool
from utils.configue import Configure
from utils.cascade_dataset import CascadeDataset
from utils.cascade_trainer import CascadeSeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
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

    model = utils.tool.get_model("unified.finetune").from_pretrained("t5-base")
    model.encoder2=copy.deepcopy(model.encoder)
    model.policy="sepenc"
    model.do_non_auto = model.do_ff_argmax = False
    model_tokenizer = AutoTokenizer.from_pretrained("t5-base",use_fast=False)
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

    # seq2seq_train_dataset=seq2seq_train_dataset[:10]
    # seq2seq_eval_dataset=seq2seq_eval_dataset[:10]
    # seq2seq_test_dataset=seq2seq_test_dataset[:10]

    targs=Args()
    targs.input_max_length=1024
    targs.generation_max_length=128
    # We wrap the "string" seq2seq data into "tokenized tensor".
    test_dataset = CascadeDataset(args, targs, model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None

    model.load_state_dict(torch.load(os.environ["CKPT"], map_location="cpu"), strict=True)
    model.eval()
    model.cuda()

    if "FROM" in os.environ:
        test_dataset.last_predictions=torch.load(os.environ["FROM"])
    predictions=torch.load(os.environ["TO"])
    atts=[]
    with torch.no_grad():
        for input,pred in zip(tqdm(test_dataset),predictions):
            del input['labels']
            input["decoder_input_ids"]=pred[:-1]
            for k,v in list(input.items()):
                input[k]=torch.tensor(v).unsqueeze(0).cuda()
            input['output_attentions']=True
            att=model(**input).cross_attentions
            att=(sum(att)/len(att)).mean(0).mean(0)
            tgt_len=(pred!=0).sum()
            src_len1=input['attention_mask'].sum().item()
            src_len2=(input['last_predictions'] != 0).sum().item()
            att=att[:tgt_len].mean(0)
            att1=att[:src_len1].mean().item()
            att2=att[1024+1:1024+1+src_len2].mean().item()
            atts.append((att1,att2))
    atts=np.float32(atts)
    torch.save(atts,os.environ["TO"]+'.att')

if __name__ == "__main__":
    main()
