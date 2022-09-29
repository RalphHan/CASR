import setproctitle

setproctitle.setproctitle('SKG')
import logging
import os
from tqdm import tqdm

import torch
import collections
if int(torch.__version__.split('.')[1]) >= 8:
    torch._six.container_abcs=collections.abc
import datasets
from transformers import (
    set_seed,
)
import utils.tool
import random
import numpy as np
from utils.cascade_dataset_sudoku import CascadeDatasetSudoku

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

    set_seed(2)

    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path='csv',
                                                                     data_files={
                                                                         k: os.path.join("./data/sudoku/",
                                                                                         f'sudoku_{k}.csv') for k in
                                                                         ['train', 'eval', 'test']})

    model = utils.tool.get_model('unified.scratch')("base")

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = [raw_datasets_split[k] for k in
                                                                         ['train', 'eval', 'test']]

    # seq2seq_train_dataset,seq2seq_eval_dataset,seq2seq_test_dataset=[[x[i] for i in range(10)] for x in [seq2seq_train_dataset,seq2seq_eval_dataset,seq2seq_test_dataset]]

    test_dataset = CascadeDatasetSudoku(seq2seq_test_dataset, False)

    model.load_state_dict(torch.load(os.environ["CKPT"], map_location="cpu"), strict=True)
    model.eval()
    model.cuda()

    if "FROM" in os.environ:
        test_dataset.last_predictions=torch.load(os.environ["FROM"])

    predictions=torch.load(os.environ["TO"])
    slice = [96339]
    with torch.no_grad():
        for order in slice:
            input = test_dataset[order]
            pred=predictions[order]
            del input['labels']
            input["decoder_input_ids"]=pred
            for k,v in list(input.items()):
                input[k]=torch.tensor(v).unsqueeze(0).cuda()
            input['output_attentions']=True
            att = model.model(
                return_dict=True,
                **input
            ).cross_attentions

            att=(sum(att)/len(att)).mean(0).mean(0).cpu().numpy()
            torch.save(att,os.environ["TO"]+'.%d.att'%order)

if __name__ == "__main__":
    main()


'''
CASTEP=4
LAST=3
CKPT=output/bart_base_sudoku_bs64/cas_${CASTEP}/checkpoint-10000/pytorch_model.bin FROM=output/bart_base_sudoku_bs64/cas_${LAST}/cas_test_generation.pk TO=output/bart_base_sudoku_bs64/cas_${CASTEP}/cas_test_generation.pk python -m empirical.attention_sudoku
'''