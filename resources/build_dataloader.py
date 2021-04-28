import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class text_dataset(Dataset):
    def __init__(self, df, label_columns, tokenizer, max_token_len):
        self.data = df
        self.label_columns = label_columns
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row["TEXT"]
        labels = data_row[self.label_columns]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
        )


def build_dataloader(df, label_columns, tokenizer, max_token_len, trainset=False):
    dataset = text_dataset(df, label_columns, tokenizer, max_token_len)

    if trainset:
        sampler = RandomSampler(df)
    else:
        sampler = SequentialSampler(df)

    return DataLoader(dataset, batch_size=10, sampler=sampler)
