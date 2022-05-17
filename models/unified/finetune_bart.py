#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BartForConditionalGeneration,BartModel
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput
from torch.nn import CrossEntropyLoss
import os

from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    add_end_docstrings,
    add_code_sample_docstrings,

)
from transformers.models.bart.modeling_bart import BART_INPUTS_DOCSTRING,_CONFIG_FOR_DOC,BART_GENERATION_EXAMPLE,shift_tokens_right,_TOKENIZER_FOR_DOC,_CHECKPOINT_FOR_DOC


@add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
@add_code_sample_docstrings(
    tokenizer_class=_TOKENIZER_FOR_DOC,
    checkpoint=_CHECKPOINT_FOR_DOC,
    output_type=Seq2SeqModelOutput,
    config_class=_CONFIG_FOR_DOC,
)
def BartModel_forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mode='casdec',
        last_predictions=None,
):
    # different to other models, Bart automatically creates decoder_input_ids from
    # input_ids if no decoder_input_ids are provided
    assert mode in {'sepenc', 'fusenc', 'casdec'}
    if decoder_input_ids is None and decoder_inputs_embeds is None:
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:

        if mode == 'fusenc':
            if last_predictions is not None:
                assert inputs_embeds is None
                assert attention_mask is not None
                last_prediction_mask = (last_predictions != self.config.pad_token_id).to(attention_mask)
                input_ids = torch.cat((input_ids, last_predictions), dim=1)
                attention_mask = torch.cat((attention_mask, last_prediction_mask), dim=1)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if mode == 'sepenc':
            if last_predictions is not None:
                assert attention_mask is not None
                last_prediction_mask = (last_predictions != self.config.pad_token_id).to(attention_mask)
                encoder_outputs_lp = self.encoder2(
                    input_ids=last_predictions,
                    attention_mask=last_prediction_mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=torch.cat((encoder_outputs[0], encoder_outputs_lp[0]), dim=1))
                attention_mask = torch.cat((attention_mask, last_prediction_mask), dim=1)
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )
def BartModel_set_input_embeddings(self, value):
    self.shared = value
    self.encoder.embed_tokens = self.shared
    if hasattr(self, 'encoder2'):
        self.encoder2.embed_tokens=self.shared
    self.decoder.embed_tokens = self.shared
BartModel.forward=BartModel_forward
BartModel.set_input_embeddings=BartModel_set_input_embeddings

class Model(BartForConditionalGeneration):
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            last_predictions=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        assert self.policy in {'sepenc', 'fusenc', 'casdec'}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=self.policy,
            last_predictions=last_predictions,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @torch.no_grad()
    def generate(self,
                 input_ids,
                 attention_mask,
                 last_predictions=None,  # TODO: check whether it is not None
                 **kwargs):

        bsz = input_ids.shape[0]

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.full((bsz, 1), -100, dtype=input_ids.dtype).to(input_ids),
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
            last_prediction_mask = (last_predictions != self.config.pad_token_id).to(attention_mask)
            attention_mask = torch.cat((attention_mask, last_prediction_mask), dim=1)
        new_prediction = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            **kwargs,
        )
        return new_prediction


if __name__=='__main__':

    pass


