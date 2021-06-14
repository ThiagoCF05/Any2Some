__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2:
    def __init__(self, tokenizer_path, model_path, max_length, device, sep_token='<VERBALIZE>'):
        self.sep_token =sep_token
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.device = device
        self.max_length = max_length

    def __call__(self, intents, texts=None):
        if texts:
            # prepare input
            messages = []
            for i, intent in enumerate(intents):
                msg = ' '.join([intent, self.sep_token, self.tokenizer.bos_token, texts[i], self.tokenizer.eos_token])
                messages.append(msg)
            # tokenize
            model_inputs = self.tokenizer(messages, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(messages, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids.to(self.device)
            # Predict
            output = self.model(**model_inputs, labels=labels) # forward pass
        else:
            # prepare input
            messages = []
            for i, intent in enumerate(intents):
                msg = ' '.join([intent, self.sep_token, self.tokenizer.bos_token])
                messages.append(msg)
            # tokenize
            model_inputs = self.tokenizer(messages, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            # Predict
            generated_ids = self.model.generate(**model_inputs, 
                                                max_length=self.max_length, 
                                                pad_token_id=self.tokenizer.pad_token_id, 
                                                eos_token_id=self.tokenizer.eos_token_id, 
                                                bos_token_id=self.tokenizer.bos_token_id)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            output = [w.split(self.sep_token)[-1] for w in output]
        return output