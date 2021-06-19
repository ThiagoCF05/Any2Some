__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelWithLMHead

class GPorTugueseGen:
    '''
    Implementation of GPorTuguese, a GPT-2 model adapted to Portuguese and based on the transformers library of HuggingFace

    Notes:
        https://huggingface.co/pierreguillou/gpt2-small-portuguese
        https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787 
    '''
    def __init__(self, tokenizer_path, model_path, max_length, device, sep_token='<VERBALIZE>'):
        '''
        params:
        ---
            tokenizer_path: path to the tokenizer in HuggingFace (e.g., pierreguillou/gpt2-small-portuguese)
            model_path: path to the model in HuggingFace (e.g., pierreguillou/gpt2-small-portuguese)
            max_length: maximum size of subtokens in the input and output
            device: cpu or gpu
            sep_token: special token to separate the meaning representation and text in order to prime the verbalization
        '''
        self.sep_token =sep_token
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelWithLMHead.from_pretrained(model_path).to(device)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
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