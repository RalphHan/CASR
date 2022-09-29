#!/usr/bin/env bash
declare -A MAX_LENGTH
MAX_LENGTH=([src]=1024 [tgt]=128)

for split in train valid test
do
    for f in src tgt
    do
        python -m fairseq.examples.roberta.multiprocessing_bpe_encoder \
         --encoder-json data/levenshtein/encoder.json --vocab-bpe data/levenshtein/vocab.bpe \
         --inputs data/levenshtein/${TASK}/${split}.${f} --outputs data/levenshtein/${TASK}/${split}.bpe.${f} --workers 1 --keep-empty
        python baselines/levenshtein/truncation.py data/levenshtein/${TASK}/${split}.bpe.${f} ${MAX_LENGTH[$f]}
     done
done

fairseq-preprocess \
  --source-lang src \
  --target-lang tgt \
  --trainpref data/levenshtein/${TASK}/train.bpe \
  --validpref data/levenshtein/${TASK}/valid.bpe \
  --testpref data/levenshtein/${TASK}/test.bpe \
  --destdir data/levenshtein/${TASK}/bin \
  --workers 1 \
  --srcdict data/levenshtein/dict.txt \
  --tgtdict data/levenshtein/dict.txt
