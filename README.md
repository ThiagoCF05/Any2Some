# Any2Some

This is a framework that aims to simplify the development of neural, end-to-end, data-to-text systems.

# Supported Models

The framework supported the following models:

### Monolingual

1. [BERT](https://huggingface.co/transformers/model_doc/bert.html)
2. [BART](https://huggingface.co/transformers/model_doc/bart.html)
3. [T5](https://huggingface.co/transformers/model_doc/t5.html)
4. [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
5. [Blenderbot](https://huggingface.co/transformers/model_doc/blenderbot.html)

### Multilingual

1. [mBERT](https://huggingface.co/bert-base-multilingual-cased) and [BERTimbau](https://github.com/neuralmind-ai/portuguese-bert/)
2. [mBART-50](https://huggingface.co/transformers/model_doc/mbart.html)
3. [mT5](https://huggingface.co/transformers/model_doc/t5.html)
4. [GPorTuguese](https://huggingface.co/pierreguillou/gpt2-small-portuguese)

# Usage

## Training

```
if [ ! -d "env" ];
then
  virtualenv env
  . env/bin/activate
  pip3 install -r requirements.txt
else
  . env/bin/activate
fi

python3 train.py --tokenizer facebook/bart-large \
                --model facebook/bart-large \
                --src_train 'example/trainsrc.txt' \
                --trg_train 'example/traintrg.txt' \
                --src_dev 'example/devsrc.txt' \
                --trg_dev 'example/devtrg.txt' \
                --epochs 3 \
                --learning_rate 1e-5 \
                --batch_size 8 \
                --early_stop 2 \
                --max_length 180 \
                --write_path bart \
                --language portuguese \
                --verbose \
                --batch_status 16 \
                --cuda
```

## Evaluation

```
. env/bin/activate

if [ ! -d "results" ];
then
  mkdir results
fi

python3 evaluate.py --tokenizer facebook/bart-large \
                --model bart/model \
                --src_test 'example/testsrc.txt' \
                --trg_test 'example/testtrg.txt' \
                --batch_size 4 \
                --max_length 180 \
                --write_dir results \
                --language portuguese \
                --verbose \
                --batch_status 16 \
                --cuda
```
