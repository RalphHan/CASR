#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput,Seq2SeqModelOutput
from torch.nn import CrossEntropyLoss
import os

from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.t5.modeling_t5 import T5_INPUTS_DOCSTRING,_CONFIG_FOR_DOC,T5LayerFF

def T5LayerFF_forward(self, hidden_states):
    # many t5/mt5 models are trained in bfloat16 and don't do well under mixed precision (fp16).
    # It appears that it's enough to disable autocast for this FF layer to avoid inf/nan
    # problems for the whole model
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
T5LayerFF._forward=T5LayerFF.forward
T5LayerFF.forward=T5LayerFF_forward

class MyT5ForConditionalGeneration(T5ForConditionalGeneration):

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
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
        mode='casdec',
        last_predictions=None,
        **kwargs
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        """
        assert mode in {'sepenc', 'fusenc', 'casdec'}
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
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
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            drop_slice = lm_logits.view(-1, lm_logits.size(-1)).sum(-1)
            drop_slice = drop_slice.isnan() | drop_slice.isinf()
            new_label = labels.view(-1).clone()
            new_label[drop_slice] = -100
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), new_label)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        if hasattr(self,'encoder2'):
            self.encoder2.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

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

class Model(MyT5ForConditionalGeneration):
    def forward(self,
                **kwargs,
                ):
        if self.do_non_auto:
            kwargs["decoder_input_ids"]=torch.full_like(kwargs["labels"],5)
        outputs = super().forward(
            mode=self.policy,
            **kwargs
        )
        if self.do_ff_argmax:
            preds=outputs.logits.argmax(-1).detach()
            return {'loss':outputs.loss,
                    'logits':preds}
        return outputs


if __name__=='__main__':

    pretrain_model = Model.from_pretrained(
        't5-base'
    )
    adapter_config = ADAPTER_CONFIG_MAP['mam']
    pretrain_model.add_adapter('adapter1', config=adapter_config)
    pretrain_model.add_adapter('adapter2', config=adapter_config)
    pretrain_model.train_adapter(['adapter1', 'adapter2'])
    pretrain_model.set_active_adapters('adapter1')
    # pretrain_model.load_adapter('haha/adapter1')
    # pretrain_model.load_adapter('haha/adapter2')
    # pretrain_model.save_all_adapters('haha')


    # print(pretrain_model.base_model.model_frozen)
    print(getattr(pretrain_model, "active_adapters", None))
    # print([x.requires_grad for x in pretrain_model.lm_head.parameters()])
    print(sum([x.requires_grad for x in pretrain_model.parameters()]))
    # pretrain_model.save_all_adapters('haha',with_head=False)
    # # pretrain_model.save_adapter('haha','my_adapter1',with_head=False)
    # from transformers.adapters.composition import AdapterCompositionBlock, Fuse
    # print(
    #         isinstance(pretrain_model.active_adapters, Fuse)
    #         or isinstance(pretrain_model.active_adapters, AdapterCompositionBlock)
    #         and any([isinstance(child, Fuse) for child in pretrain_model.active_adapters.children])
    # )
    # print(pretrain_model.tokenizer)
    # print(sum([param.nelement() if param.requires_grad else 0 for param in pretrain_model.parameters()]))
    # keys1=pretrain_model.state_dict().keys()
    # adapter_config = ADAPTER_CONFIG_MAP['mam']
    # add a fresh adapter
    # pretrain_model.add_adapter('my_adapter', config=adapter_config)
    # # Freeze all model weights except of those of this adapter
    # pretrain_model.train_adapter(['my_adapter'])
    # # Set the adapters to be used in every forward pass
    # pretrain_model.set_active_adapters('my_adapter')
    # keys2=pretrain_model.state_dict().keys()
    # state=pretrain_model.state_dict()
    # print(len(keys1),len(keys2))
    # print(sum([len(state[x].ravel()) for x in keys1]))
    # print(sum([state[x].nelement() for x in keys2-keys1]))
    # # print(keys2-keys1)
    # print(sum([param.nelement() for param in pretrain_model.parameters()]))
    # print([param[0] for param in pretrain_model.named_parameters() if param[1].requires_grad])
    # print(keys2-keys1-set([param[0] for param in pretrain_model.named_parameters() if param[1].requires_grad]))


