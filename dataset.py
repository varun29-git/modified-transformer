import torch
from torch.utils.data import Dataset
import tiktoken


class LanguageModelDataset(Dataset):
    def __init__(self, texts, seq_len, model_name="gpt-4"):
        
        self.texts = texts
        self.seq_len = seq_len

        # GPT-4 tokenizer
        self.tokenizer = tiktoken.encoding_for_model(model_name)

        # GPT tokenizers do not have PAD, so we define one
        self.pad_id = self.tokenizer.eot_token  # safe choice

        # Pre-tokenize everything once 
        self.tokens = []
        for text in texts:
            self.tokens.extend(self.tokenizer.encode(text))

    def __len__(self):
        # Number of chunks
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        # Pad if needed 
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.pad_id] * (self.seq_len + 1 - len(chunk))

        chunk = torch.tensor(chunk, dtype=torch.long)

    
        input_ids = chunk[:-1]
        targets = chunk[1:]

        return {
            "input_ids": input_ids,     # (T,)
            "targets": targets          # (T,)
        }
