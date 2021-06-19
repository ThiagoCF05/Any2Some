if [ ! -d "env" ];
then
  virtualenv env
  . env/bin/activate
  pip3 install -r requirements.txt
else
  . env/bin/activate
fi

# BART
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