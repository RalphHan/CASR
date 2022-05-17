#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
import math

class Prefix(nn.Module):
    def __init__(self, preseqlen, n_embd, mid_dim, match_n_layer, match_n_head, match_n_embd, prefix_dropout):
        super().__init__()
        self.wte = nn.Embedding(preseqlen, n_embd)
        self.match_n_layer = match_n_layer
        self.match_n_head = match_n_head
        self.match_n_embd = match_n_embd
        self.control_trans = nn.Sequential(
            nn.Linear(n_embd, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, match_n_layer * 2 * n_embd),
        )
        self.dropout = nn.Dropout(prefix_dropout)

    def forward(self, input_tokens, description, sample_size):
        temp_control = self.wte(input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values, seqlen


class VersionPrefix(nn.Module):
    def __init__(self, max_versions, freeze_prefix, preseqlen, n_embd, mid_dim, match_n_layer, match_n_head,
                 match_n_embd, prefix_dropout):
        super().__init__()
        self.max_versions = max_versions
        self.freeze_prefix = freeze_prefix
        self.version = -1
        self.prefixes = nn.ModuleList(
            Prefix(preseqlen, n_embd, mid_dim, match_n_layer, match_n_head, match_n_embd, prefix_dropout)
            for i in range(max_versions))
        for param in self.prefixes.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_version(self):
        version = self.version
        assert version + 1 < self.max_versions
        if version + 1 > 0:
            for param_old, param_new in zip(self.prefixes[version].parameters(),
                                            self.prefixes[version + 1].parameters()):
                param_new.data.copy_(param_old.data)
                param_old.requires_grad = False

        if not self.freeze_prefix:
            for param_new in self.prefixes[version + 1].parameters():
                param_new.requires_grad = True

        self.version = version + 1

    # @torch.no_grad()
    # def set_version(self,version):
    #     '''
    #     For evaluate only
    #     :param version:
    #     :return:
    #     '''
    #     assert 0<=version <self.max_versions
    #     self.version=version

    def forward(self, *args, **kwargs):
        assert self.version >= 0
        return self.prefixes[self.version](*args, **kwargs)


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self._keys_to_ignore_on_save = []
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.register_buffer('middle_prompt',
                             torch.LongTensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('->'))]))

        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location
        )
        self.config = self.pretrain_model.config
        from ..prompt.modeling_bart import BartForConditionalGeneration
        from ..prompt.modeling_t5 import T5ForConditionalGeneration
        if isinstance(self.pretrain_model, BartForConditionalGeneration):
            self.match_n_layer = self.config.decoder_layers
            self.match_n_head = self.config.decoder_attention_heads
        elif isinstance(self.pretrain_model, (T5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
        else:
            raise ValueError("Other models are not supported yet!")

        self.n_embd = self.config.d_model
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())
        self.prefix: VersionPrefix = None
        self.prefix_enc: VersionPrefix = None
        self.prefix_dec: VersionPrefix = None
        for name in ['prefix', 'prefix_enc', 'prefix_dec']:
            setattr(self, name,
                    VersionPrefix(args.max_cascade_steps, self.args.model.freeze_prefix,
                                  self.preseqlen, self.n_embd,
                                  self.mid_dim, self.match_n_layer, self.match_n_head, self.match_n_embd,
                                  args.prefix_tuning.prefix_dropout))

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {}
        for name in ['prefix', 'prefix_enc', 'prefix_dec']:
            state_dict.update({f'{name}.{key}': value for key, value in getattr(self, name).state_dict().items()})
        return state_dict

    @torch.no_grad()
    def update_version(self):
        for name in ['prefix','prefix_enc','prefix_dec']:
            getattr(self, name).update_version()

    # @torch.no_grad()
    # def set_version(self, version):
    #     '''
    #     For evaluate only
    #     :param version:
    #     :return:
    #     '''
    #     for name in ['prefix', 'prefix_dec']:
    #         getattr(self, name).set_version(version)
    #     self.prefix_enc.set_version(0)

    def get_prompt(self, bsz=None, sample_size=1, description=None):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        past_key_values, _ = self.prefix(input_tokens, description, sample_size)
        past_key_values_dec, _ = self.prefix_dec(input_tokens, description, sample_size)

        # Encoder prefix
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        past_key_values_enc, seqlen = self.prefix_enc(input_tokens_enc, description, 1)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(old_bsz, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result

    @torch.no_grad()
    def get_description_representation(self, kwargs):
        if self.args.model.use_description and self.args.model.map_description:
            description_input_ids = kwargs.pop("description_input_ids")
            description_attention_mask = kwargs.pop("description_attention_mask")
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                last_predictions=None,  # TODO: check whether it is not None
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation,
        )

        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
            mode='fusenc',
            last_predictions=last_predictions,
        )
        loss=outputs.loss
        # if math.isnan(loss.item()):
        #     print(labels)
        return {'loss': loss}

    @torch.no_grad()
    def generate(self,
                 input_ids,
                 attention_mask,
                 last_predictions=None,  # TODO: check whether it is not None
                 **kwargs):

        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'], description=description_representation,
        )
        one_past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation,
        )
        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.full((bsz,1), -100, dtype=input_ids.dtype).to(input_ids),
            past_prompt=one_past_prompt,
            mode='fusenc',
            last_predictions=last_predictions,
            output_attentions=True,
            output_hidden_states=True,
        )
        encoder_outputs = BaseModelOutput(
            last_hidden_state=outputs.encoder_last_hidden_state,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
        )
        if last_predictions is not None:
            last_prediction_mask = (last_predictions != self.pretrain_model.config.pad_token_id).to(attention_mask)
            attention_mask = torch.cat((attention_mask, last_prediction_mask), dim=1)
        new_prediction = self.pretrain_model.generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            past_prompt=past_prompt,
            mode='fusenc',
            **kwargs,
        )
        return new_prediction

