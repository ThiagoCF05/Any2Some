__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BartTokenizer, BartForConditionalGeneration

class BARTGen:
    def __init__(self, tokenizer_path, model_path, max_length, device, multilingual, src_lang='', trg_lang=''):
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