from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from tag_compound.load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=1, shuffle=False)
    model.eval()
    output_pred = []

    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)
            output_pred += list(result)
    
    return output_pred

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_from_disk(dataset_dir)['question']
    test_label = [0] * len(test_dataset)
    
    # tokenize dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def main(question, checkpoint=700):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load tokenizer
    MODEL_NAME = 'kykim/bert-kor-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load model
    MODEL_DIR = f"./tag_compound/results/{checkpoint}"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    # load test datset
    test_dataset, test_label = tokenized_test = tokenized_dataset(question, tokenizer)

    # predict answer
    pred_answer = inference(model, test_dataset, device)
    
    # return predicted answer
    label_decoder = {0: '[WHO]', 1: '[WHEN]', 2: '[WHERE]', 3: '[WHAT]', 4: '[HOW]', 5: '[WHY]', 6: '[QUANTITY]', 7: '[CITE]'}
    return label_decoder[pred_answer]

  