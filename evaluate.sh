. env/bin/activate

if [ ! -d "results" ];
then
  mkdir results
fi

# webnlg
python3 evaluate.py --tokenizer facebook/bart-large \
                --model bart \
                --src_test 'example/testsrc.txt' \
                --trg_test 'example/testtrg.txt' \
                --batch_size 4 \
                --max_length 180 \
                --write_dir results \
                --language portuguese \
                --verbose \
                --batch_status 16 \
                --cuda