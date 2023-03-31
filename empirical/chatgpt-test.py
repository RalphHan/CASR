import setproctitle
setproctitle.setproctitle('SKG')
import logging
import os
os.sys.path.insert(0,'')
import random
from dotenv import load_dotenv
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import torch
import collections
if int(torch.__version__.split('.')[1]) >= 8:
    torch._six.container_abcs=collections.abc
import datasets
import utils.tool
from utils.configue import Configure
from tqdm import tqdm
import pickle as pk
import numpy as np

prefix_prompt={
    'webqsp':
        "WebQSP is a classic dataset for KBQA(Knowledge Base Question Answering). "
        "The input consists of a knowledge graph and an NL query, and the output is an s-Expression which can be executed on the knowledge graph. "
        ,
    'mtop':
        "MTOP is a benchmark for comprehensive multilingual task-oriented semantic parsing. "
        "The input consists of a list of API calls and an NL query, and the output is a tree-based TOP Representation that can be executed."
        ,
    'kvret':
        "KVRET is a benchmark for table conversation. "
        "The input consists of a table and an NL query, and the output is an NL response corresponding to the dialog. "
        ,
    'sudoku':
        "Sudoku is an open dataset on Kaggle. "
        "Its game target is to fill the blanks correctly with the constraint that any two numbers in the same row, column, and house shouldn’t have the same value. "
        "The input is a flattened sudoku game board with 9x9=81 numbers, where 0 denote blanks to be solved, and non-zero positions cannot be modified. "
        "The output is also with 81 numbers where blanks are filled with the correct values. "
}

mid_prompt='''
For each query, you should give the answer without any explanation or any additional information. 
When a suggested answer is given (may not be correct), you should repeat it if it's correct, or correct it if it's wrong. 
For example,

Query: <query1>
Suggested Answer: None
Answer: <answer1>

Query: <query2>
Suggested Answer: <suggested_answer2>
Answer: 
'''

def answer(task,query1,answer1,query2,suggested_answer2=None):
    prompt=prefix_prompt[task]+mid_prompt.replace('<query1>',query1).\
                                            replace('<answer1>',answer1).\
                                            replace('<query2>',query2).\
                                            replace('<suggested_answer2>',suggested_answer2 or 'None')
    complete=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    return complete

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)

def get_knowledge(raw_item,args,conv_sep = " || "):
    if raw_item["text_in"]:
        ###################
        # With text input #
        ###################
        if conv_sep in raw_item["text_in"]:
            ##################
            # Conversational #
            ##################
            # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
            # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
            index = raw_item["text_in"].index(conv_sep)
            if args.model.knowledge_usage == 'concatenate' or args.model.knowledge_usage is None:
                # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                              raw_item["struct_in"],
                                                                              raw_item["text_in"][
                                                                              index + len(conv_sep):])
            elif args.model.knowledge_usage == 'separate':
                # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                   raw_item["text_in"][index + len(conv_sep):])
            else:
                raise ValueError()
        else:
            ######################
            # Non-conversational #
            ######################
            if args.model.knowledge_usage == 'concatenate' or args.model.knowledge_usage is None:
                # seq_in  = "text_in ; structured knowledge: struct_in"
                seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
            elif args.model.knowledge_usage == 'separate':
                # seq_in  = "text_in"
                seq_in = raw_item["text_in"]
            else:
                raise ValueError()
    else:
        ######################
        # Without text input #
        ######################
        if args.model.knowledge_usage == 'concatenate':
            # seq_in  = "structured knowledge: struct_in"
            seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
        elif args.model.knowledge_usage == 'separate':
            # seq_in  = ""
            seq_in = ""
        else:
            raise ValueError()

    # Concatenate description.
    if args.model.use_description and args.model.concatenate_description:
        seq_in = "{} ; {}".format(raw_item["description"], seq_in)
    return seq_in

def deal_sudoku() -> None:
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path='csv',
                                                                     data_files={'test': 'data/sudoku/sudoku_test.csv'})

    seq2seq_test_dataset = raw_datasets_split['test']
    random.seed(12321)
    slice1 = random.sample(range(len(seq2seq_test_dataset)), 100)
    slice2 = random.sample(sorted(set(range(len(seq2seq_test_dataset))) - set(slice1)), 100)
    slice = slice2 + slice1

    query1, answer1 = seq2seq_test_dataset[slice[0]]['quizzes'], seq2seq_test_dataset[slice[0]]['solutions']
    os.makedirs(f'output/{os.environ["TASK"]}', exist_ok=True)
    st=-1
    for i in range(1,len(slice)):
        if not os.path.exists(f'output/{os.environ["TASK"]}/{slice[i]}.txt'):
            st=max(1,i-1)
            break

    for x in tqdm(slice[st:]):
        query2 = seq2seq_test_dataset[x]['quizzes']
        answer2 = seq2seq_test_dataset[x]['solutions']
        suggested_answer2 = None
        with open(f'output/{os.environ["TASK"]}/{x}.txt', 'w') as f:
            f.write('input:' + query2 + '\n')
            f.write('gt:' + answer2 + '\n')
            for castep in range(5):
                given_answer2 = answer(os.environ['TASK'], query1, answer1, query2, suggested_answer2)
                f.write(f'{castep}:' + given_answer2 + '\n')
                suggested_answer2 = given_answer2
    targets = [seq2seq_test_dataset[x] for x in slice[1:]]
    results = [[] for _ in range(5)]
    for x in slice[1:]:
        with open(f'output/{os.environ["TASK"]}/{x}.txt', 'r') as f:
            lines = f.readlines()
            for castep in range(5):
                given_answer2 = lines[castep+2].split(':')[1].strip()
                results[castep].append(given_answer2)
    metric = []
    for i in range(5):
        correct_num,tot_num=0,0
        for pred,gt in zip(results[i],targets):
            try:
                quizzes=np.int64(list(gt["quizzes"]))
                solutions=np.int64(list(gt["solutions"]))
                tot_num+=(quizzes==0).sum()
                pred = np.int64(list(pred))
                assert len(pred)==len(solutions)
                correct_num+=((quizzes==0)&(pred==solutions)).sum()
            except:
                pass
        metric.append(correct_num/tot_num)
    print(metric)
    with open(f'output/{os.environ["TASK"]}/metric.pk', 'wb') as f:
        pk.dump(metric, f)
def deal_huggingface() -> None:
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

    if len(seq2seq_dataset_split) == 3:
        _, _, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")
    random.seed(12321)
    slice1 = random.sample(range(len(seq2seq_test_dataset)), 100)
    slice2 = random.sample(sorted(set(range(len(seq2seq_test_dataset)))-set(slice1)),100)
    slice=slice2+slice1

    query1, answer1=get_knowledge(seq2seq_test_dataset[slice[0]],args),seq2seq_test_dataset[slice[0]]['seq_out']
    os.makedirs(f'output/{os.environ["TASK"]}', exist_ok=True)
    st = -1
    for i in range(1, len(slice)):
        if not os.path.exists(f'output/{os.environ["TASK"]}/{slice[i]}.txt'):
            st = max(1, i - 1)
            break

    for x in tqdm(slice[st:]):
        query2 = get_knowledge(seq2seq_test_dataset[x], args)
        answer2 = seq2seq_test_dataset[x]['seq_out']
        suggested_answer2 = None
        with open(f'output/{os.environ["TASK"]}/{x}.txt', 'w') as f:
            f.write('input:' + query2 + '\n')
            f.write('gt:' + answer2 + '\n')
            for castep in range(3):
                given_answer2=answer(os.environ['TASK'],query1,answer1,query2,suggested_answer2)
                f.write(f'{castep}:' + given_answer2 + '\n')
                suggested_answer2=given_answer2
    targets=[seq2seq_test_dataset[x] for x in slice[1:]]
    results = [[] for _ in range(3)]
    for x in slice[1:]:
        with open(f'output/{os.environ["TASK"]}/{x}.txt', 'r') as f:
            lines = f.readlines()
            for castep in range(3):
                given_answer2 = lines[castep + 2].split(':',1)[1].strip()
                results[castep].append(given_answer2)
    metric=[]
    for i in range(3):
        metric.append(evaluator.evaluate(results[i],targets, "test"))
    print(metric)
    with open(f'output/{os.environ["TASK"]}/metric.pk', 'wb') as f:
        pk.dump(metric, f)


if __name__ == "__main__":
   if os.environ['TASK']=='sudoku':
       deal_sudoku()
   else:
       deal_huggingface()

# webqsp 1639 340581 1.021743
# mtop 4386 902221 2.7066630000000003
# kvret 808 216189 0.648567

