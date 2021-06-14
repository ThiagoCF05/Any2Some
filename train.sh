if [ ! -d "logs" ];
then
  mkdir logs
fi

if [ ! -d "env" ];
then
  virtualenv env
  . env/bin/activate
  pip3 install torch
  pip3 install transformers
  pip3 install nltk
else
  . env/bin/activate
fi

# BART
python3 train.py facebook/bart-large facebook/bart-large \
                '../data/writer/train.err.tok' \
                '../data/writer/train.cor.tok' \
                '../data/writer/dev.err.tok' \
                '../data/writer/dev.cor.tok' \
                3 \
                1e-5 \
                8 \
                2 \
                128 \
                logs/bart \
                english \
                --verbose \
                --batch_status 128 \
                --cuda