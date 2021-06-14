. env/bin/activate

if [ ! -d "results" ];
then
  mkdir results
fi

# webnlg
python3 evaluate.py facebook/bart-large logs/bart \
                '../data/writer/dev.src' \
                '../data/writer/dev.trg' \
                4 \
                512 \
                results/bart \
                english \
                --verbose \
                --batch_status 16 \
                --cuda