__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BartTokenizer, BartForConditionalGeneration

class BARTGen:
    '''
    Implementation of BART and mBART models based on the transformers library of HuggingFace

    Notes:
        https://huggingface.co/transformers/model_doc/bart.html
        https://huggingface.co/transformers/model_doc/mbart.html
    '''
    def __init__(self, tokenizer_path, model_path, max_length, device, multilingual, src_lang='', trg_lang=''):
        '''
        params:
        ---
            tokenizer_path: path to the tokenizer in HuggingFace (e.g., facebook/bart-large)
            model_path: path to the model in HuggingFace (e.g., facebook/bart-large)
            max_length: maximum size of subtokens in the input and output
            device: cpu or gpu
            multilingual: is the model multilingual? True or False
            src_lang: if the model is multilingual, set the language of the source tokenizer (see https://huggingface.co/transformers/model_doc/mbart.html)
            trg_lang: if the model is multilingual, set the language of the target tokenizer (see https://huggingface.co/transformers/model_doc/mbart.html)
        '''
        if multilingual:
            assert src_lang != ''
            self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_path, src_lang=src_lang, tgt_lang=trg_lang)
            self.model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
        else:
            self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
            self.model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length

    def __call__(self, intents, texts=None):
        '''
        Method that convert a meaning representation into text (e.g. intents)

        params:
        ---
            intents: list of input meaning representations (strings)
            texts: list of output gold-standard verbalizations
        
        return:
        ---
            output: during training (texts not None), returns the list of probabilities. 
                Otherwise, returns the predicted verbalizations to the input meaning representations
        '''
        # tokenize
        model_inputs = self.tokenizer(intents, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        # Predict
        if texts:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids.to(self.device)
            # Predict
            output = self.model(**model_inputs, labels=labels) # forward pass
        else:
            generated_ids = self.model.generate(**model_inputs, max_length=self.max_length)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output