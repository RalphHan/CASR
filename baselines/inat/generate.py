from fairseq_cli.generate import cli_main as fairseq_generate
import shlex
import os
import sys
import torch
torch_embedding=torch.embedding
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    indices_clone=indices.clone()
    indices_clone[(indices>=weight.size(0))&(indices!=padding_idx)]=weight.size(0)-1
    return torch_embedding(weight,indices_clone,padding_idx,scale_grad_by_freq,sparse)

torch.embedding=embedding

cmd=f"""
fairseq-generate \
    data/levenshtein/{os.environ["TASK"]}/bin/ \
    --gen-subset test \
    --task translation_lev \
    --path {os.environ["output_dir"]}/checkpoint_best.pt \
    --iter-decode-max-iter 3 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --bpe gpt2 \
    --print-step \
    --truncate-source \
    --nbest 1 \
    --source-lang src \
    --target-lang tgt \
    --batch-size 1 \
    --max-source-positions 1026 \
    --max-target-positions 130 \
    --results-path {os.environ["output_dir"]}/result
"""
sys.argv = shlex.split(cmd)
fairseq_generate()