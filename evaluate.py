__author__='thiagocastroferreira'

import os
import json
import argparse
from models.bartgen import BARTGen
from models.bert import BERTGen
from models.gportuguesegen import GPorTugueseGen
from models.t5gen import T5Gen
from models.gpt2 import GPT2
from models.blenderbot import Blenderbot
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class Inferencer:
    def __init__(self, model, testdata, batch_status, device, write_dir, verbose=True, language='portuguese'):
        '''
        params:
        ---
            model: model to be trained
            test: test data
            batch_status: update the loss after each 'batch_status' updates
            device: cpu or gpy
            write_dir: folder to save results
            verbose
            language
        '''
        self.model = model
        self.batch_status = batch_status
        self.device = device
        self.verbose = verbose
        self.testdata = testdata
        self.write_dir = write_dir
        self.language = language
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
    
    
    def evaluate(self):
        self.model.model.eval()
        results = {}
        for batch_idx, inp in enumerate(self.testdata):
            intent, text = inp['X'], inp['y']
            if intent not in results:
                results[intent] = { 'idx': batch_idx, 'intent': intent, 'hyp': '', 'refs': [] }
                # predict
                output = self.model([intent])
                results[intent]['hyp'] = output[0]

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx+1, \
                        len(self.testdata), 100. * batch_idx / len(self.testdata)))
            
            results[intent]['refs'].append(text)
        
        results = sorted(results.values(), key=lambda x: x['idx'])
        path = os.path.join(self.write_dir, 'data.txt')
        with open(path, 'w') as f:
            f.write('\n'.join([w['intent'] for w in results]))
        
        path = os.path.join(self.write_dir, 'output.txt')
        with open(path, 'w') as f:
            f.write('\n'.join([w['hyp'] for w in results]))
        
        path = os.path.join(self.write_dir, 'result.json')
        json.dump(results, open(path, 'w'), separators=(',', ':'), sort_keys=True, indent=4)

        hyps, refs = [], []
        for i, row in enumerate(results):
            if self.language != 'english':
                hyps.append(nltk.word_tokenize(row['hyp'], language=self.language))
                refs.append([nltk.word_tokenize(ref, language=self.language) for ref in row['refs']])
            else:
                hyps.append(nltk.word_tokenize(row['hyp']))
                refs.append([nltk.word_tokenize(ref) for ref in row['refs']])
        
        chencherry = SmoothingFunction()
        bleu = corpus_bleu(refs, hyps, smoothing_function=chencherry.method3)
        print('BLEU: ', bleu)
        return bleu

class NewsDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (string): data
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def load_data(src_fname, trg_fname):
    with open(src_fname) as f:
        src = f.read().split('\n')
    with open(trg_fname) as f:
        trg = f.read().split('\n') 
    
    assert len(src) == len(trg)
    data = [{ 'X': src[i], 'y': trg[i] } for i in range(len(src))]
    return data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", help="path to the tokenizer", required=True)
    parser.add_argument("--model", help="path to the model", required=True)
    parser.add_argument("--src_test", help="path to the source dev data", required=True)
    parser.add_argument("--trg_test", help="path to the target dev data", required=True)
    parser.add_argument("--batch_size", help="batch size of test", type=int, default=16)
    parser.add_argument("--max_length", help="maximum length to be processed by the network", type=int, default=180)
    parser.add_argument("--write_dir", help="path to write results", required=True)
    parser.add_argument("--language", help="language", default='english')
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--src_lang", help="source language of mBART tokenizer", default='pt_XX')
    parser.add_argument("--trg_lang", help="target language of mBART tokenizer", default='pt_XX')
    args = parser.parse_args()
    # settings
    batch_size = args.batch_size
    batch_status = args.batch_status
    language = args.language
    try:
        verbose = args.verbose
    except:
        verbose = False
    try:
        device = 'cuda' if args.cuda else 'cpu'
    except:
        device = 'cpu'
    write_dir = args.write_dir

    # model
    max_length = args.max_length
    tokenizer_path = args.tokenizer
    model_path = args.model
    if 'mbart' in tokenizer_path:
        src_lang = args.src_lang
        trg_lang = args.trg_lang
        generator = BARTGen(tokenizer_path, model_path, max_length, device, True, src_lang, trg_lang)
    elif 'bart' in tokenizer_path:
        generator = BARTGen(tokenizer_path, model_path, max_length, device, False)
    elif 'bert' in tokenizer_path:
        generator = BERTGen(tokenizer_path, model_path, max_length, device)
    elif 'mt5' in tokenizer_path:
        generator = T5Gen(tokenizer_path, model_path, max_length, device, True)
    elif 't5' in tokenizer_path:
        generator = T5Gen(tokenizer_path, model_path, max_length, device, False)
    elif 'gpt2-small-portuguese' in tokenizer_path:
        generator = GPorTugueseGen(tokenizer_path, model_path, max_length, device)
    elif tokenizer_path == 'gpt2':
        generator = GPT2(tokenizer_path, model_path, max_length, device)
    elif 'blenderbot' in tokenizer_path:
        generator = Blenderbot(tokenizer_path, model_path, max_length, device)
    else:
        raise Exception("Invalid model") 

    # data
    # test data
    src_fname = args.src_test
    trg_fname = args.trg_test
    testdata = load_data(src_fname, trg_fname)

    inf = Inferencer(generator, testdata, batch_status, device, write_dir, verbose, language)
    inf.evaluate()