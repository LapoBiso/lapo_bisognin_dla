from datasets import load_dataset, get_dataset_split_names
import numpy as np
import collections
import torch

def load_data(name):
    """
    load dataset splits
    """
    splits = get_dataset_split_names(name)
    ds_dict = {split: load_dataset(name, split = split) for split in splits}
    return ds_dict

def set_classifier_data(ds_dict, feats):
    """
    concatenate CLS token features and list labels
    """
    feats_concatenated = torch.vstack([feat[0][0] for feat in feats]) 
    labels = list(ds_dict['label'])
    return feats_concatenated, labels

def set_tokenized_data(ds, tokenizer):
    """
    tokenize dataset text data
    """
    tokenized = ds.map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True)
    return tokenized

def process_multimodal_data(ds, processor):
    processed = ds.map(lambda batch: processor(batch['caption'], batch['image']))
    return processed