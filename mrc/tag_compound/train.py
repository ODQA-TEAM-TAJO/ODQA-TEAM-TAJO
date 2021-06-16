import pickle as pickle
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers import AdamW, get_linear_schedule_with_warmup
from tag_compound.load_data import *
from tag_compound.tag_compound import *
from importlib import import_module
from sklearn.model_selection import train_test_split
import argparse


# metrics function for evaluation
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

# set fixed random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def train():
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    MODEL_NAME = 'kykim/bert-kor-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = pd.read_csv("./tag_compound/data/tag_train_augmented.tsv", header=None, sep="\t")
    valid_dataset = load_from_disk('./data/train_dataset/validation')

    train_tag = load_dataset("./tag_compound/data/tag_train_augmented.tsv")
    valid_tag = load_dataset("./tag_compound/data/tag_valid.tsv")

    train_x = list(train_dataset.iloc[:,0])
    train_y = list(train_tag.iloc[:,-1])
    val_x = valid_dataset['question']
    val_y = list(valid_tag.iloc[:,-1])
    
    # tokenize datasets
    tokenized_train = tokenized_dataset(train_x, tokenizer)
    tokenized_val = tokenized_dataset(val_x, tokenizer)

    # make dataset for pytorch
    RE_train_dataset = RE_Dataset(tokenized_train, train_y)
    RE_valid_dataset = RE_Dataset(tokenized_val, val_y)

    # instantiate pretrained language model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)

    # set arguments for trainer
    training_args = TrainingArguments(
        output_dir="./tag_compound/results",
        logging_dir="./tag_compound/logs",
        logging_steps=100,
        save_total_limit=1,
        evaluation_strategy='steps',
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=True,

        dataloader_num_workers=4,
        seed=42,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        label_smoothing_factor=0.3,
        learning_rate=3e-5,
        warmup_steps=300,
        weight_decay=0.001,
    )

    trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset= RE_valid_dataset,             # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,         # define metrics function
    )

    # train model
    trainer.train()



