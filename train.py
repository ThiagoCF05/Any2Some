__author__='thiagocastroferreira'

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
nltk.download('punkt')
import os
import torch
from torch import optim

class Trainer:
    '''
    Module for training a generative neural model
    '''
    def __init__(self, model, trainloader, devdata, optimizer, epochs, \
        batch_status, device, write_path, early_stop=5, verbose=True, language='english'):
        '''
        params:
        ---
            model: model to be trained
            trainloader: training data
            devdata: dev data
            optimizer
            epochs: number of epochs
            batch_status: update the loss after each 'batch_status' updates
            device: cpu or gpy
            write_path: folder to save best model
            early_stop
            verbose
            language
        '''
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_status = batch_status
        self.device = device
        self.early_stop = early_stop
        self.verbose = verbose
        self.trainloader = trainloader
        self.devdata = devdata
        self.write_path = write_path
        self.language = language
        if not os.path.exists(write_path):
            os.mkdir(write_path)
    
    def train(self):
        '''
        Train model based on the parameters specified in __init__ function
        '''
        max_bleu, repeat = 0, 0
        for epoch in range(self.epochs):
            self.model.model.train()
            losses = []
            for batch_idx, inp in enumerate(self.trainloader):
                intents, texts = inp['X'], inp['y']
                self.optimizer.zero_grad()

                # generating
                output = self.model(intents, texts)

                # Calculate loss
                loss = output.loss
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(epoch, \
                        batch_idx+1, len(self.trainloader), 100. * batch_idx / len(self.trainloader), 
                        float(loss), round(sum(losses) / len(losses), 5)))
            
            bleu, acc = self.evaluate()
            checkpoint = { 'epoch': epoch+1, 'bleu': bleu, 'acc': acc, 'best_model': False }
            print('BLEU: ', bleu, 'Accuracy: ', acc)
            if bleu > max_bleu:
                self.model.model.save_pretrained(os.path.join(self.write_path, 'model'))
                max_bleu = bleu
                repeat = 0
                checkpoint['best_model'] = True
                print('Saving best model...')
            else:
                repeat += 1
            
            if repeat == self.early_stop:
                break
            
            # saving checkpoint
            if os.path.exists(f"{self.write_path}/checkpoint.json"):
                checkpoints = json.load(open(f"{self.write_path}/checkpoint.json"))
                checkpoints['checkpoints'].append(checkpoint)
            else:
                checkpoints = { 'checkpoints': [checkpoint] }
            json.dump(checkpoints, open(f"{self.write_path}/checkpoint.json", 'w'), separators=(',', ':'), sort_keys=True, indent=4)
    
    def evaluate(self):
        '''
        Evaluating the model in devset after each epoch
        '''
        self.model.model.eval()
        results = {}
        for batch_idx, inp in enumerate(self.devdata):
            intent, text = inp['X'], inp['y']
            if intent not in results:
                results[intent] = { 'hyp': '', 'refs': [] }
                # predict
                output = self.model([intent])
                results[intent]['hyp'] = output[0]

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx+1, \
                        len(self.devdata), 100. * batch_idx / len(self.devdata)))
            
            results[intent]['refs'].append(text)
        
        hyps, refs, acc = [], [], 0
        for i, intent in enumerate(results.keys()):
            if i < 20 and self.verbose:
                print('Real: ', results[intent]['refs'][0])
                print('Pred: ', results[intent]['hyp'])
                print()
            
            if self.language != 'english':
                hyps.append(nltk.word_tokenize(results[intent]['hyp'], language=self.language))
                refs.append([nltk.word_tokenize(ref, language=self.language) for ref in results[intent]['refs']])
            else:
                hyps.append(nltk.word_tokenize(results[intent]['hyp']))
                refs.append([nltk.word_tokenize(ref) for ref in results[intent]['refs']])
            
            if results[intent]['hyp'] in results[intent]['refs'][0]:
                acc += 1
        
        chencherry = SmoothingFunction()
        bleu = corpus_bleu(refs, hyps, smoothing_function=chencherry.method3)
        return bleu, float(acc) / len(results)
    
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
    parser.add_argument("--src_train", help="path to the source train data", required=True)
    parser.add_argument("--trg_train", help="path to the target train data", required=True)
    parser.add_argument("--src_dev", help="path to the source dev data", required=True)
    parser.add_argument("--trg_dev", help="path to the target dev data", required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=5)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--early_stop", help="earling stop", type=int, default=3)
    parser.add_argument("--max_length", help="maximum length to be processed by the network", type=int, default=180)
    parser.add_argument("--write_path", help="path to write best model", required=True)
    parser.add_argument("--language", help="language", default='english')
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--src_lang", help="source language of mBART tokenizer", default='pt_XX')
    parser.add_argument("--trg_lang", help="target language of mBART tokenizer", default='pt_XX')
    args = parser.parse_args()
    # settings
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    batch_status = args.batch_status
    early_stop =args.early_stop
    language = args.language
    try:
        verbose = args.verbose
    except:
        verbose = False
    try:
        device = 'cuda' if args.cuda else 'cpu'
    except:
        device = 'cpu'
    write_path = args.write_path

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

    # train data
    src_fname = args.src_train
    trg_fname = args.trg_train
    data = load_data(src_fname, trg_fname)
    dataset = NewsDataset(data)
    trainloader = DataLoader(dataset, batch_size=batch_size)
    
    # dev data
    src_fname = args.src_dev
    trg_fname = args.trg_dev
    devdata = load_data(src_fname, trg_fname)

    # optimizer
    optimizer = optim.AdamW(generator.model.parameters(), lr=learning_rate)
    
    # trainer
    trainer = Trainer(generator, trainloader, devdata, optimizer, epochs, batch_status, device, write_path, early_stop, verbose, language)
    trainer.train()