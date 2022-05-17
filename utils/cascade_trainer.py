import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import NamedTuple

import numpy as np
import torch
import transformers.trainer_seq2seq
from packaging import version
from torch import nn
from torch.utils.data import Dataset
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PredictionOutput, speed_metrics
import os
from .training_arguments import WrappedSeq2SeqTrainingArguments

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]


class CascadeSeq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
            self,
            evaluator,
            args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            test_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.test_dataset = test_dataset
        self.test_examples = test_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        if self.args.do_non_auto:
            self.args.predict_with_generate = False
        self.model.do_non_auto=self.args.do_non_auto

    def train_all(self):
        # _path='output/sepenc_T5_base_finetune_fetaqa/cas_0/'
        # for _mode in ['train','eval','test']:
        #     getattr(self,f'{_mode}_dataset').last_predictions = torch.load(os.path.join(_path,f'cas_{_mode}_generation.pk'))
        self.origin_output_dir = self.args.output_dir
        if self.args.do_restart and self.is_world_process_zero():
            torch.save(self.model.state_dict(),os.path.join(self.origin_output_dir, 'init.pt'))
            assert not hasattr(self.model,'update_version')
        torch.distributed.barrier()
        for cascade_step in range(self.args.max_cascade_steps):
            if self.args.do_restart and cascade_step>0:
                self.model.load_state_dict(torch.load(os.path.join(self.origin_output_dir, 'init.pt'),map_location='cpu'))
            if hasattr(self.model,'update_version'):
                self.model.update_version()
            self.args.output_dir = os.path.join(self.origin_output_dir, 'cas_%d' % cascade_step)
            os.makedirs(self.args.output_dir, exist_ok=True)
            self.model.do_ff_argmax = False
            if cascade_step>0 or not self.args.start_from_first_castep:
                self.optimizer = None
                self.lr_scheduler = None
                self.train()
                self.save_state()
            else:
                max_steps = self.args.max_steps
                self.args.max_steps = 1
                state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.train()
                self.args.max_steps = max_steps
                self.model.load_state_dict(state_dict, strict=False)
                del state_dict
            self.model.do_ff_argmax = self.args.do_non_auto
            if cascade_step!=self.args.max_cascade_steps-1 and not self.args.do_origin:
                self.update_dataset(self.train_dataset, None, metric_key_prefix='cas_train', with_metric=False,
                                cascade_step=cascade_step)
            self.update_dataset(self.eval_dataset, self.eval_examples, metric_key_prefix='cas_eval', with_metric=True,
                                cascade_step=cascade_step)
            self.update_dataset(self.test_dataset, self.test_examples, metric_key_prefix='cas_test', with_metric=True,
                                cascade_step=cascade_step)
        self.args.output_dir = self.origin_output_dir
        self.save_state()


    def update_dataset(self, dataset, examples, metric_key_prefix, with_metric, cascade_step):
        if dataset is None:
            return
        output = self.predict(dataset, examples, metric_key_prefix=metric_key_prefix, with_metric=with_metric)
        if not self.args.do_origin:
            dataset.last_predictions = output.predictions
        if self.is_world_process_zero():
            torch.save(output.predictions, os.path.join(self.args.output_dir, f'{metric_key_prefix}_generation.pk'))
        if with_metric:
            metrics = output.metrics
            max_samples = len(dataset)
            metrics["samples"] = min(max_samples, len(dataset))
            split_name = f'{metric_key_prefix}_{cascade_step}'
            self.log_metrics(split_name, metrics)
            output_dir = self.args.output_dir
            self.args.output_dir = os.path.join(self.origin_output_dir, 'results')
            os.makedirs(self.args.output_dir, exist_ok=True)
            self.save_metrics(split_name, {split_name: metrics})
            self.args.output_dir = output_dir

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "predict",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
            with_metric=True,
    ) -> PredictionOutput:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.args.do_non_auto:
            for prediction in output.predictions:
                _st=np.where(prediction == 1)[0]
                if len(_st)==0:
                    prediction[-1]=1
                else:
                    prediction[_st.min() + 1:] = 0
        if not with_metric:
            self._memory_tracker.stop_and_update_metrics(output.metrics)
            return output
        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                test_examples, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds, section="test"))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "no_repeat_ngram_size": 0,  # FIXME: hard coding the no_repeat_ngram_size
        }

        if "description_input_ids" in inputs:
            gen_kwargs["description_input_ids"] = inputs["description_input_ids"]
        if "description_attention_mask" in inputs:
            gen_kwargs["description_attention_mask"] = inputs["description_attention_mask"]
        if "task_ids" in inputs:
            gen_kwargs["task_ids"] = inputs["task_ids"]
        if "last_predictions" in inputs:
            gen_kwargs["last_predictions"]=inputs["last_predictions"]

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        return (None, generated_tokens, None)

    def _post_process_function(
            self, examples: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        # assert isinstance(examples, Dataset)
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        return EvalPrediction(predictions=predictions, items=[examples[idx] for idx in range(len(predictions))])

    def _compute_metrics(self, eval_prediction: EvalPrediction, section) -> dict:
        return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, section)
