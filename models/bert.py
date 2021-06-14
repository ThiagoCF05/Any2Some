__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import EncoderDecoderModel, BertTokenizer

class BERTGen:
    def __init__(self, tokenizer_path, model_path, max_length, device):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        try:
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, model_path).to(device)
        except:
            self.model = EncoderDecoderModel.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length

    def __call__(self, intents, texts=None):
        # tokenize
        model_inputs = self.tokenizer(intents, truncation=True, padding=True, add_special_tokens=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        # Predict
        if texts:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids.to(self.device)
            # Predict
            output = self.model(**model_inputs, decoder_input_ids=labels, labels=labels) # forward pass
        else:
            generated_ids = self.model.generate(**model_inputs, max_length=self.max_length, decoder_start_token_id=self.model.config.decoder.pad_token_id)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output