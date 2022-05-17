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
import numpy as np
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
import utils.tool
from utils.configue import Configure
from utils.cascade_dataset import CascadeDataset
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

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

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
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
    print(seq2seq_test_dataset[0]['seq_out'])
    length=[len(x['seq_out']) for x in seq2seq_test_dataset]
    print(length[:10])
    for cas_step in range(1,3):
        if os.path.exists(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step-1}/cas_test_lang.pk'):
            predictions = torch.load(
                f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step-1}/cas_test_lang.pk')
        else:
            predictions=torch.load(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step-1}/cas_test_generation.pk')
            predictions=model_tokenizer.batch_decode(predictions, skip_special_tokens=True)
            torch.save(predictions,f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step-1}/cas_test_lang.pk')
        atts=torch.load(f'output/sepenc_T5_base_finetune_{os.environ["TASK"]}_continue/cas_{cas_step}/cas_test_generation.pk.att')
        print('cas_step:',cas_step)
        # print('refine dominatye ratio:',(atts[:,0]<atts[:,1]).sum()/atts.shape[0])
        print('avg att:',atts[:,0].mean(),atts[:,1].mean())
        st_len = sorted(atts[:,1])
        pt1 = st_len[int(len(st_len) * 0.33)]
        pt2 = st_len[int(len(st_len) * 0.67)]
        print('short:',evaluator.evaluate([x for i,x in enumerate(predictions) if atts[i,1]<=pt1], [x for i,x in enumerate(seq2seq_test_dataset) if atts[i,1]<=pt1], "test"))
        print('avg len:',np.mean([x for i,x in enumerate(length) if atts[i,1]<=pt1]))
        print('middle:',evaluator.evaluate([x for i,x in enumerate(predictions) if pt1<atts[i,1]<=pt2], [x for i,x in enumerate(seq2seq_test_dataset) if pt1<atts[i,1]<=pt2], "test"))
        print('avg len:', np.mean([x for i, x in enumerate(length) if pt1<atts[i,1]<=pt2]))
        print('long:',evaluator.evaluate([x for i,x in enumerate(predictions) if pt2<atts[i,1]], [x for i,x in enumerate(seq2seq_test_dataset) if pt2<atts[i,1]], "test"))
        print('avg len:', np.mean([x for i, x in enumerate(length) if pt2<atts[i,1]]))

if __name__ == "__main__":
    main()

# webqsp
# cas_step: 1
# avg att: 0.0009997181 0.0033606538
# short: {'META_TUNING/webqsp.cfg/F1': 0.3959933745883232, 'avr': 0.3959933745883232}
# avg len: 129.80591497227357
# middle: {'META_TUNING/webqsp.cfg/F1': 0.8782105452905342, 'avr': 0.8782105452905342}
# avg len: 68.17204301075269
# long: {'META_TUNING/webqsp.cfg/F1': 0.8450139538951028, 'avr': 0.8450139538951028}
# avg len: 51.57962962962963
# cas_step: 2
# avg att: 0.00089938345 0.0060602473
# short: {'META_TUNING/webqsp.cfg/F1': 0.5188250390905513, 'avr': 0.5188250390905513}
# avg len: 132.22181146025878
# middle: {'META_TUNING/webqsp.cfg/F1': 0.8583813626477313, 'avr': 0.8583813626477313}
# avg len: 66.28673835125448
# long: {'META_TUNING/webqsp.cfg/F1': 0.8578216771222059, 'avr': 0.8578216771222059}
# avg len: 51.10740740740741


# mtop
# cas_step: 1
# avg att: 0.0010574334 0.0032790927
# short: {'META_TUNING/mtop.cfg/exact_match': 0.6954419889502762, 'META_TUNING/mtop.cfg/template_accuracy': 0.7520718232044199, 'avr': 0.7237569060773481}
# avg len: 106.32527624309392
# middle: {'META_TUNING/mtop.cfg/exact_match': 0.7987927565392354, 'META_TUNING/mtop.cfg/template_accuracy': 0.8417169684775319, 'avr': 0.8202548625083836}
# avg len: 70.32461435278337
# long: {'META_TUNING/mtop.cfg/exact_match': 0.8645473393227368, 'META_TUNING/mtop.cfg/template_accuracy': 0.8804422944022114, 'avr': 0.8724948168624741}
# avg len: 41.45058742225294
# cas_step: 2
# avg att: 0.001020566 0.004305966
# short: {'META_TUNING/mtop.cfg/exact_match': 0.7272099447513812, 'META_TUNING/mtop.cfg/template_accuracy': 0.7852209944751382, 'avr': 0.7562154696132597}
# avg len: 106.24447513812154
# middle: {'META_TUNING/mtop.cfg/exact_match': 0.8316566063044937, 'META_TUNING/mtop.cfg/template_accuracy': 0.8752515090543259, 'avr': 0.8534540576794098}
# avg len: 70.13615023474179
# long: {'META_TUNING/mtop.cfg/exact_match': 0.8762957843814789, 'META_TUNING/mtop.cfg/template_accuracy': 0.8908085694540429, 'avr': 0.8835521769177609}
# avg len: 41.725639253628195

#kvret
# cas_step: 1
# avg att: 0.003991912 0.008419272
# short: {'META_TUNING/kvret.cfg/bleu': 0.16552187828938156, 'META_TUNING/kvret.cfg/all_micro': 0.6104725415070243, 'META_TUNING/kvret.cfg/all_macro': 0.5923118528171419, 'META_TUNING/kvret.cfg/navigate_micro': 0.6021505376344086, 'META_TUNING/kvret.cfg/navigate_macro': 0.5738543262141215, 'META_TUNING/kvret.cfg/weather_micro': 0.6229508196721312, 'META_TUNING/kvret.cfg/weather_macro': 0.6397908016099471, 'META_TUNING/kvret.cfg/schedule_micro': 0.5942028985507246, 'META_TUNING/kvret.cfg/schedule_macro': 0.5614582813731339, 'avr': 0.5514126597408905}
# avg len: 67.75280898876404
# middle: {'META_TUNING/kvret.cfg/bleu': 0.2015131168117852, 'META_TUNING/kvret.cfg/all_micro': 0.7331887201735358, 'META_TUNING/kvret.cfg/all_macro': 0.6804651693770022, 'META_TUNING/kvret.cfg/navigate_micro': 0.6644518272425248, 'META_TUNING/kvret.cfg/navigate_macro': 0.6102880650901792, 'META_TUNING/kvret.cfg/weather_micro': 0.7106227106227107, 'META_TUNING/kvret.cfg/weather_macro': 0.6636363623351159, 'META_TUNING/kvret.cfg/schedule_micro': 0.8103448275862069, 'META_TUNING/kvret.cfg/schedule_macro': 0.8113174586948253, 'avr': 0.6539809175482095}
# avg len: 49.56
# long: {'META_TUNING/kvret.cfg/bleu': 0.20948064925493762, 'META_TUNING/kvret.cfg/all_micro': 0.8778280542986426, 'META_TUNING/kvret.cfg/all_macro': 0.7769151121826896, 'META_TUNING/kvret.cfg/schedule_micro': 0.8878048780487805, 'META_TUNING/kvret.cfg/schedule_macro': 0.7984901258108836, 'META_TUNING/kvret.cfg/navigate_micro': 0.7499999999999999, 'META_TUNING/kvret.cfg/navigate_macro': 0.5999999880000002, 'META_TUNING/kvret.cfg/weather_micro': 0, 'META_TUNING/kvret.cfg/weather_macro': 0.0, 'avr': 0.5445020897328816}
# avg len: 26.026315789473685
# cas_step: 2
# avg att: 0.0038645084 0.010135514
# short: {'META_TUNING/kvret.cfg/bleu': 0.16701155390224637, 'META_TUNING/kvret.cfg/all_micro': 0.6417445482866044, 'META_TUNING/kvret.cfg/all_macro': 0.6097608589310116, 'META_TUNING/kvret.cfg/navigate_micro': 0.6013745704467354, 'META_TUNING/kvret.cfg/navigate_macro': 0.5679709605539291, 'META_TUNING/kvret.cfg/weather_micro': 0.6630286493860846, 'META_TUNING/kvret.cfg/weather_macro': 0.6764578837760705, 'META_TUNING/kvret.cfg/schedule_micro': 0.6689655172413793, 'META_TUNING/kvret.cfg/schedule_macro': 0.6369609295921009, 'avr': 0.5814750524573513}
# avg len: 69.02621722846442
# middle: {'META_TUNING/kvret.cfg/bleu': 0.20175612135849408, 'META_TUNING/kvret.cfg/all_micro': 0.7509025270758123, 'META_TUNING/kvret.cfg/all_macro': 0.7297161167483506, 'META_TUNING/kvret.cfg/navigate_micro': 0.702928870292887, 'META_TUNING/kvret.cfg/navigate_macro': 0.6783068772302007, 'META_TUNING/kvret.cfg/weather_micro': 0.7357142857142858, 'META_TUNING/kvret.cfg/weather_macro': 0.719873662282085, 'META_TUNING/kvret.cfg/schedule_micro': 0.8012820512820513, 'META_TUNING/kvret.cfg/schedule_macro': 0.814285712435065, 'avr': 0.6816406916021368}
# avg len: 44.694545454545455
# long: {'META_TUNING/kvret.cfg/bleu': 0.20264357985381942, 'META_TUNING/kvret.cfg/all_micro': 0.8580246913580247, 'META_TUNING/kvret.cfg/all_macro': 0.7606462574167636, 'META_TUNING/kvret.cfg/schedule_micro': 0.8812260536398467, 'META_TUNING/kvret.cfg/schedule_macro': 0.7949047603149522, 'META_TUNING/kvret.cfg/navigate_micro': 0.7586206896551724, 'META_TUNING/kvret.cfg/navigate_macro': 0.6944444405864196, 'META_TUNING/kvret.cfg/weather_micro': 0.8, 'META_TUNING/kvret.cfg/weather_macro': 0.4999999750000013, 'avr': 0.6945011608694445}
# avg len: 29.778195488721803
