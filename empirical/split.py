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
    print(f'{os.environ["TASK"]} {len(seq2seq_train_dataset)} {len(seq2seq_eval_dataset)} {len(seq2seq_test_dataset)}')


if __name__ == "__main__":
    main()
# webqsp
# cas_step: 0
# short: {'META_TUNING/webqsp.cfg/F1': 0.7879588795673328, 'avr': 0.7879588795673328}
# middle: {'META_TUNING/webqsp.cfg/F1': 0.8307219967212881, 'avr': 0.8307219967212881}
# long: {'META_TUNING/webqsp.cfg/F1': 0.5050979113038168, 'avr': 0.5050979113038168}
# cas_step: 1
# short: {'META_TUNING/webqsp.cfg/F1': 0.8012115014884276, 'avr': 0.8012115014884276}
# middle: {'META_TUNING/webqsp.cfg/F1': 0.8640553300546213, 'avr': 0.8640553300546213}
# long: {'META_TUNING/webqsp.cfg/F1': 0.5751947256258185, 'avr': 0.5751947256258185}
# cas_step: 2
# short: {'META_TUNING/webqsp.cfg/F1': 0.8012115014884276, 'avr': 0.8012115014884276}
# middle: {'META_TUNING/webqsp.cfg/F1': 0.8640553300546213, 'avr': 0.8640553300546213}
# long: {'META_TUNING/webqsp.cfg/F1': 0.5812872493763348, 'avr': 0.5812872493763348}


# mtop
# cas_step: 0
# short: {'META_TUNING/mtop.cfg/exact_match': 0.8365253077975376, 'META_TUNING/mtop.cfg/template_accuracy': 0.8522571819425444, 'avr': 0.844391244870041}
# middle: {'META_TUNING/mtop.cfg/exact_match': 0.8051247471341875, 'META_TUNING/mtop.cfg/template_accuracy': 0.8482805124747134, 'avr': 0.8267026298044504}
# long: {'META_TUNING/mtop.cfg/exact_match': 0.7161693268563497, 'META_TUNING/mtop.cfg/template_accuracy': 0.7730742539902845, 'avr': 0.7446217904233171}
# cas_step: 1
# short: {'META_TUNING/mtop.cfg/exact_match': 0.8549931600547196, 'META_TUNING/mtop.cfg/template_accuracy': 0.872093023255814, 'avr': 0.8635430916552668}
# middle: {'META_TUNING/mtop.cfg/exact_match': 0.830074173971679, 'META_TUNING/mtop.cfg/template_accuracy': 0.8718813216453135, 'avr': 0.8509777478084963}
# long: {'META_TUNING/mtop.cfg/exact_match': 0.7494795281054824, 'META_TUNING/mtop.cfg/template_accuracy': 0.8070784177654406, 'avr': 0.7782789729354616}
# cas_step: 2
# short: {'META_TUNING/mtop.cfg/exact_match': 0.8625170998632011, 'META_TUNING/mtop.cfg/template_accuracy': 0.8796169630642955, 'avr': 0.8710670314637483}
# middle: {'META_TUNING/mtop.cfg/exact_match': 0.8334457181389077, 'META_TUNING/mtop.cfg/template_accuracy': 0.8752528658125421, 'avr': 0.854349291975725}
# long: {'META_TUNING/mtop.cfg/exact_match': 0.7536433032616239, 'META_TUNING/mtop.cfg/template_accuracy': 0.8126301179736294, 'avr': 0.7831367106176266}

#kvret
# cas_step: 0
# short: {'META_TUNING/kvret.cfg/bleu': 0.14046348866111125, 'META_TUNING/kvret.cfg/all_micro': 0.6590909090909092, 'META_TUNING/kvret.cfg/all_macro': 0.6470899440085663, 'META_TUNING/kvret.cfg/navigate_micro': 0.6666666666666667, 'META_TUNING/kvret.cfg/navigate_macro': 0.6743055513411458, 'META_TUNING/kvret.cfg/weather_micro': 0, 'META_TUNING/kvret.cfg/weather_macro': 0.0, 'META_TUNING/kvret.cfg/schedule_micro': 0.64, 'META_TUNING/kvret.cfg/schedule_macro': 0.5599999888000001, 'avr': 0.4430685053964888}
# middle: {'META_TUNING/kvret.cfg/bleu': 0.2192518170870105, 'META_TUNING/kvret.cfg/all_micro': 0.7115839243498818, 'META_TUNING/kvret.cfg/all_macro': 0.6540005936470806, 'META_TUNING/kvret.cfg/schedule_micro': 0.7179487179487178, 'META_TUNING/kvret.cfg/schedule_macro': 0.707596370197333, 'META_TUNING/kvret.cfg/navigate_micro': 0.7045454545454545, 'META_TUNING/kvret.cfg/navigate_macro': 0.6261603367601342, 'META_TUNING/kvret.cfg/weather_micro': 0.7126436781609194, 'META_TUNING/kvret.cfg/weather_macro': 0.6531936802980661, 'avr': 0.634102730332733}
# long: {'META_TUNING/kvret.cfg/bleu': 0.15084561287168577, 'META_TUNING/kvret.cfg/all_micro': 0.6569014084507043, 'META_TUNING/kvret.cfg/all_macro': 0.6407851183512832, 'META_TUNING/kvret.cfg/navigate_micro': 0.583941605839416, 'META_TUNING/kvret.cfg/navigate_macro': 0.5507453411359974, 'META_TUNING/kvret.cfg/weather_micro': 0.6118721461187214, 'META_TUNING/kvret.cfg/weather_macro': 0.6461897187768824, 'META_TUNING/kvret.cfg/schedule_micro': 0.7789473684210527, 'META_TUNING/kvret.cfg/schedule_macro': 0.7730471791802895, 'avr': 0.599252833238448}
# cas_step: 1
# short: {'META_TUNING/kvret.cfg/bleu': 0.12171000688380007, 'META_TUNING/kvret.cfg/all_micro': 0.6593406593406593, 'META_TUNING/kvret.cfg/all_macro': 0.6597883566465105, 'META_TUNING/kvret.cfg/navigate_micro': 0.6666666666666667, 'META_TUNING/kvret.cfg/navigate_macro': 0.6993055511848958, 'META_TUNING/kvret.cfg/weather_micro': 0, 'META_TUNING/kvret.cfg/weather_macro': 0.0, 'META_TUNING/kvret.cfg/schedule_micro': 0.64, 'META_TUNING/kvret.cfg/schedule_macro': 0.5333333226666669, 'avr': 0.4422382848210222}
# middle: {'META_TUNING/kvret.cfg/bleu': 0.19805464126885106, 'META_TUNING/kvret.cfg/all_micro': 0.7151095732410611, 'META_TUNING/kvret.cfg/all_macro': 0.6877571073853476, 'META_TUNING/kvret.cfg/schedule_micro': 0.7183673469387756, 'META_TUNING/kvret.cfg/schedule_macro': 0.7280045334140482, 'META_TUNING/kvret.cfg/navigate_micro': 0.7063197026022305, 'META_TUNING/kvret.cfg/navigate_macro': 0.6536467743383613, 'META_TUNING/kvret.cfg/weather_micro': 0.7195467422096317, 'META_TUNING/kvret.cfg/weather_macro': 0.7034496742255351, 'avr': 0.6478062328470935}
# long: {'META_TUNING/kvret.cfg/bleu': 0.16330995058980177, 'META_TUNING/kvret.cfg/all_micro': 0.6947835738068812, 'META_TUNING/kvret.cfg/all_macro': 0.6656115257574351, 'META_TUNING/kvret.cfg/navigate_micro': 0.6029411764705882, 'META_TUNING/kvret.cfg/navigate_macro': 0.5710835053695198, 'META_TUNING/kvret.cfg/weather_micro': 0.6646616541353383, 'META_TUNING/kvret.cfg/weather_macro': 0.6772968622052729, 'META_TUNING/kvret.cfg/schedule_micro': 0.8128161888701517, 'META_TUNING/kvret.cfg/schedule_macro': 0.8000370963528377, 'avr': 0.628060170395314}
# cas_step: 2
# short: {'META_TUNING/kvret.cfg/bleu': 0.12322416375999659, 'META_TUNING/kvret.cfg/all_micro': 0.6451612903225806, 'META_TUNING/kvret.cfg/all_macro': 0.6470899440085665, 'META_TUNING/kvret.cfg/navigate_micro': 0.6470588235294117, 'META_TUNING/kvret.cfg/navigate_macro': 0.6826388846223959, 'META_TUNING/kvret.cfg/weather_micro': 0, 'META_TUNING/kvret.cfg/weather_macro': 0.0, 'META_TUNING/kvret.cfg/schedule_micro': 0.64, 'META_TUNING/kvret.cfg/schedule_macro': 0.5333333226666669, 'avr': 0.4353896032121798}
# middle: {'META_TUNING/kvret.cfg/bleu': 0.19770250671467915, 'META_TUNING/kvret.cfg/all_micro': 0.7178899082568807, 'META_TUNING/kvret.cfg/all_macro': 0.6894302090575448, 'META_TUNING/kvret.cfg/schedule_micro': 0.7183673469387756, 'META_TUNING/kvret.cfg/schedule_macro': 0.7280045334140482, 'META_TUNING/kvret.cfg/navigate_micro': 0.7007299270072993, 'META_TUNING/kvret.cfg/navigate_macro': 0.651356237873509, 'META_TUNING/kvret.cfg/weather_micro': 0.7308781869688386, 'META_TUNING/kvret.cfg/weather_macro': 0.711113364689751, 'avr': 0.6494969134357029}
# long: {'META_TUNING/kvret.cfg/bleu': 0.16998057702332103, 'META_TUNING/kvret.cfg/all_micro': 0.6918378678511936, 'META_TUNING/kvret.cfg/all_macro': 0.6681686177871761, 'META_TUNING/kvret.cfg/navigate_micro': 0.6096654275092935, 'META_TUNING/kvret.cfg/navigate_macro': 0.5815596958366006, 'META_TUNING/kvret.cfg/weather_micro': 0.6356821589205397, 'META_TUNING/kvret.cfg/weather_macro': 0.65940879792058, 'META_TUNING/kvret.cfg/schedule_micro': 0.8288590604026846, 'META_TUNING/kvret.cfg/schedule_macro': 0.8056761940897789, 'avr': 0.6278709330379075}