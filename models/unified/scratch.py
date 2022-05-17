#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import BartForConditionalGeneration,BartConfig

from .base import PushToHubFriendlyModel


MODEL_CONFIG={
    'small':BartConfig(
            vocab_size=19+3,
            max_position_embeddings=81+1,
            encoder_layers=3,
            encoder_ffn_dim=512,
            encoder_attention_heads=4,
            decoder_layers=3,
            decoder_ffn_dim=512,
            decoder_attention_heads=4,
            d_model=128,
        ),
    'base':BartConfig(
            vocab_size=19+3,
            max_position_embeddings=81+1,
            encoder_layers=6,
            encoder_ffn_dim=2048,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            d_model=512,
    ),
}

class Model(PushToHubFriendlyModel):
    def __init__(self,size):
        super().__init__()
        self._keys_to_ignore_on_save = []
        assert size in MODEL_CONFIG
        self.config = MODEL_CONFIG[size]
        self.model = BartForConditionalGeneration(self.config)
    def forward(self,
                input_ids,
                decoder_input_ids,
                labels,
                **kwargs,
                ):

        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )
        if self.do_ff_argmax:
            preds=outputs.logits[...,13:22].argmax(-1).detach()+13
            preds[labels==-100]=input_ids[labels==-100]
            return {'loss':outputs.loss,
                    'logits':preds}
        else:
            return {'loss':outputs.loss}

    @torch.no_grad()
    def generate(self,
                 input_ids,
                 **kwargs):

        def restrict_decode_vocab(batch_idx, prefix_beam):
            length=len(prefix_beam)
            if length>=81+1:
                return []
            idx=input_ids[batch_idx][length-1].item()
            if idx>3 and idx<=9+3:
                return [idx]
            return list(range(3+9+1,19+3))

        prediction = self.model.generate(
            input_ids=input_ids,
            use_cache=True,
            prefix_allowed_tokens_fn=restrict_decode_vocab,
            **kwargs,
        )

        return prediction

if __name__=='__main__':
    gen_kwargs = {
        "max_length": 83,
        "num_beams": 4,
        "synced_gpus": False,
        "no_repeat_ngram_size": 0,  # FIXME: hard coding the no_repeat_ngram_size
    }
    model=Model()
    input_ids=torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 10, 9]*9]) + 3
    generated_tokens = model.generate(
        input_ids,
        **gen_kwargs,
    )
    print(input_ids)
    print(generated_tokens)
    print(generated_tokens.size())

    from transformers.models.bart.modeling_bart import shift_tokens_right
    labels = torch.LongTensor([[1, -103, 3, 4, -103, 6, 7, 10, 9] * 9]) + 3
    decoder_input_ids = shift_tokens_right(
        labels, 1, 2
    )
    print(decoder_input_ids)
