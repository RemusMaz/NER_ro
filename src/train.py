import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from nervaluate import Evaluator
import numpy as np



# we'll define or model name here
transformer_model_name = "dumitrescustefan/bert-base-romanian-cased-v1"


"""## Data loading

First thing we'll do, and actually half the work is to preprocess our data into a format easy for Pytorch Lightning (and us) to work with.

Let's start by reading the dataset from the json files. 
"""

with open("train.json", "r", encoding="utf8") as f:
    train_data = json.load(f)
with open("valid.json", "r", encoding="utf8") as f:
    validation_data = json.load(f)
with open("test.json", "r", encoding="utf8") as f:
    test_data = json.load(f)


# deduce bio2 tag mapping and simple tag list, required by nervaluate
tags = ["O"]*16  # tags without the B- or I- prefix
bio2tags = ["O"]*31 # tags with the B- and I- prefix, all tags are here

for instance in train_data:
    for tag, tag_index in zip(instance["ner_tags"], instance["ner_ids"]):
        bio2tags[tag_index] = tag  # put the bio2 tag in it's correct position
        if tag_index % 2 == 0 and tag_index > 0:
          tags[int(tag_index/2)] = tag[2:]

print(f"Dataset contains {len(bio2tags)} BIO2 classes: {bio2tags}.\n")
print(f"There are {len(tags)} classes: {tags}")


model = TransformerModel(
    model_name=transformer_model_name,
    lr=2e-5,
    model_max_length=512,
    bio2tag_list=bio2tags,
    tag_list=tags
)
print(model)
early_stop = EarlyStopping(
    monitor='valid/strict',
    min_delta=0.001,
    patience=5,
    verbose=True,
    mode='max'
)

trainer = pl.Trainer(
    
    devices=-1,  # uncomment this when training on gpus
    accelerator="gpu",  # uncomment this when training on gpus
    max_epochs=6,  # set this to -1 when training fully
    callbacks=[early_stop], 
    limit_train_batches=100,  # comment this out when training fully
    limit_val_batches=5,  # comment this out when training fully
    gradient_clip_val=1.0,
    enable_checkpointing=True  # this disables saving the model each epoch
)

# instantiate dataloaders
# a batch_size of 8 should work fine on 16GB GPUs
train_dataloader = DataLoader(MyDataset(tokenizer, train_data), batch_size=8, collate_fn=my_collator, shuffle=True, pin_memory=True)
validation_dataloader = DataLoader(MyDataset(tokenizer, validation_data), batch_size=8, collate_fn=my_collator, shuffle=False, pin_memory=True)

# call this to start training
trainer.fit(model, train_dataloader, validation_dataloader)

# let's test our code
# model.eval()

# test = ["George", "Alexndru", "Popescu", "merge", "cu", "trenul", "Cluj", "-", "Timișoara", "de", "ora", "6", ":", "20", "."]
# predicted_class = predict(model, test)

# for word, cls in zip(test, predicted_class):
#   print(f"{word} is a {cls}")
