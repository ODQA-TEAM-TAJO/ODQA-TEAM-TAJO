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
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent['input_ids'].to(device),
            attention_mask=tokenized_sent['attention_mask'].to(device),
            # token_type_ids=tokenized_sent['token_type_ids'].to(device)
        )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        pred = np.argmax(logits, axis=-1)
    
    return pred


def main(question, checkpoint):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load tokenizer
    MODEL_NAME = 'kykim/bert-kor-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load model
    MODEL_DIR = f"./tag_compound/results/{checkpoint}"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    # load test datset
    test_dataset = tokenized_test = tokenized_dataset(question, tokenizer)

    # predict answer
    pred_answer = inference(model, test_dataset, device)
    
    # return predicted answer
    label_decoder = {0: '[WHO]', 1: '[WHEN]', 2: '[WHERE]', 3: '[WHAT]', 4: '[HOW]', 5: '[WHY]', 6: '[QUANTITY]', 7: '[CITE]'}
    return label_decoder[pred_answer[0]]

  