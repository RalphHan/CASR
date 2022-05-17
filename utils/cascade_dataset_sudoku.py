import os
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers.models.bart.modeling_bart import shift_tokens_right
class CascadeDatasetSudoku(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self,seq2seq_dataset,do_non_auto):
        self.seq2seq_dataset = seq2seq_dataset
        self.last_predictions = None
        self.do_non_auto=do_non_auto

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        src = torch.from_numpy(np.int64(list(raw_item['quizzes']))+3)
        tgt = torch.from_numpy(np.int64(list(raw_item['solutions']))+3)
        tgt[src == 3] += 9
        if not self.do_non_auto:
            decoder_input_ids=shift_tokens_right(
                tgt.unsqueeze(0), 1, 2
            )[0]
        tgt[src != 3] = -100
        if self.last_predictions is not None:
            src = torch.from_numpy(self.last_predictions[index])
        if self.do_non_auto:
            decoder_input_ids = shift_tokens_right(
                src.unsqueeze(0), 1, 2
            )[0]
        return {'input_ids': src, 'labels': tgt, 'decoder_input_ids': decoder_input_ids}

    def __len__(self):
        return len(self.seq2seq_dataset)
