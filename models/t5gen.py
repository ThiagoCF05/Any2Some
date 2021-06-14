__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, T5Tokenizer

class T5Gen:
    def __init__(self, tokenizer_path, model_path, max_length, device, multilingual, sep_token='Verbalize:'):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        if multilingual:
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length
        self.sep_token = sep_token

    def __call__(self, intents, texts=None):
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