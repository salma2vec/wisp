from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, name):
        self.tk = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token
    def encode(self, text):
        return self.tk(text, return_tensors="pt")
    def decode(self, ids):
        return self.tk.batch_decode(ids, skip_special_tokens=True)
    @property
    def eos_token_id(self):
        return self.tk.eos_token_id
