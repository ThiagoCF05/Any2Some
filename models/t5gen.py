__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, T5Tokenizer

class T5Gen:
    '''
    Implementation of T5 and mT5 models based on the transformers library of HuggingFace

    Notes:
        https://huggingface.co/transformers/model_doc/t5.html 
        https://huggingface.co/transformers/model_doc/mt5.html 
    '''
    def __init__(self, tokenizer_path, model_path, max_length, device, multilingual, sep_token='Verbalize:'):
        '''
        params:
        ---
            tokenizer_path: path to the tokenizer in HuggingFace (e.g., facebook/bart-large)
            model_path: path to the model in HuggingFace (e.g., facebook/bart-large)
            max_length: maximum size of subtokens in the input and output
            device: cpu or gpu
            multilingual: is the model multilingual? True or False
            sep_token: special token to separate the meaning representation and text in order to prime the verbalization
        '''
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        if multilingual:
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length
        self.sep_token = sep_token

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
        # prepare
        for i, intent in enumerate(intents):
            intents[i] = ' '.join([self.sep_token, intent])
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