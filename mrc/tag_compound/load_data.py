import pickle as pickle
import os
import pandas as pd
import torch

# convert to torch Dataset
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# load tsv datasets
def load_dataset(dataset_dir):
    # load dataset
    dataset = pd.read_csv(dataset_dir, header=None, delimiter='\t')

    label_encoder = {'[WHO]': 0, '[WHEN]': 1, '[WHERE]': 2, '[WHAT]': 3, '[HOW]': 4, '[WHY]': 5, '[QUANTITY]': 6, '[CITE]': 7}
    dataset.iloc[:,-1] = dataset.iloc[:,-1].apply(lambda x: label_encoder[x])
    
    return dataset

# tokenization
def tokenized_dataset(dataset, tokenizer):    
    tokenized_sentences = tokenizer(
        dataset,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
        add_special_tokens=True,
    )
    return tokenized_sentences
