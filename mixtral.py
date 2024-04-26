#!/usr/bin/env python
# coding: utf-8

import torch
import random
from transformers import HfArgumentParser, Seq2SeqTrainingArguments,EarlyStoppingCallback

import logging

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from datasets import load_dataset, concatenate_datasets,Value
import numpy as np
from typing import Union, Optional
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, AutoModel
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,GenerationConfig
)
from arguments import ModelArguments, DataArguments
import wandb
from nltk.tokenize import sent_tokenize
import nltk
from evaluate import load

nltk.download("punkt")
logger = logging.getLogger(__name__)

import pathlib
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from evaluate import load
# from utils import get_dataset
import numpy as np
from peft import PeftModel    
import logging
import os
from torch.utils.data import DataLoader 
from tqdm import tqdm
from typing import List


import os
import re
from bisect import bisect_left
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from datasets import load_dataset, load_from_disk
import json
import random
import re
from evaluate import load

  



## Few-shots + explicit_errors
few_shots_explicit = '''You are given a few examples of erroneous sentences and their corrections. For every example, you are given the explicit errors and their corrections. You are also given an erroneous input sentence and its correction, followed by the explicit errors in this sentence. Output a simple explanation for each error in the erroneous sentence.'''

## Few shots + no explicit errors
few_shots_no_explicit = '''You are given a few examples of erroneous sentences and their corrections. You are also given an erroneous input sentence and its correction. Output a simple explanation for each error in the erroneous sentence.'''

## No few-shots + explicit
no_few_shots_explicit = '''You are given an erroneous input sentence and its correction, followed by the explicit errors in this sentence. Output a simple explanation for each error in the erroneous sentence.'''

## No few-shots + no explicit
no_few_shots_no_explicit = '''You are given an erroneous input sentence and its correction. Output a simple explanation for each error in the erroneous sentence.'''



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def put_explicit_error(example):
    ret = ''
    ret += f"Errors:\n"

    for i,correction in enumerate(example['corrections']):
        correct_word = ' ' if correction['correct'] == '-' else correction['correct']
        error_word = ' ' if correction['error'] == '-' else correction['error']

        ret += f"{i+1}. Error: {error_word}, Correction: {correct_word}\n"
    
    return ret


def generate_prompt(example, prompt_head = None,num_shots=0, explicit_errors = 0,train = 1, dataset = None, field = None):
    full_prompt  = prompt_head

    # print(example.keys())

    if num_shots:
        idx= np.random.randint(0,len(dataset),num_shots)
        samples = dataset.select(idx)

        for sample in samples:
            full_prompt += f"Erroneous sentence: {sample['incorrect_sentence']}\n"
            full_prompt += f"Correct sentence: {sample['correct_sentence']}\n"
            if explicit_errors:
                full_prompt += put_explicit_error(sample)

            full_prompt += f"Explanations:\n"
            for i,correction in enumerate(sample['corrections']):
                full_prompt += f"{i+1}. {correction['explanation']}\n"
            
    full_prompt += f"Erroneous sentence: {example['incorrect_sentence']}\n"
    full_prompt += f"Correct sentence: {example['correct_sentence']}\n"

    if explicit_errors:
        full_prompt += put_explicit_error(example)
        
    full_prompt += f"Explanations:\n"

    labels = ''
    for i,correction in enumerate(example['corrections']):
        labels += f"{i+1}. {correction['explanation']}\n"
    if train:
            full_prompt += labels


    # example['text'] = 'sd'
    example[field] = full_prompt.strip()
    example['label'] = labels
    # print(example.keys())

    return example





def get_dataset(dataset_path, split='train', field='prompt', num_shots=0,explicit_errors = 0):
    

    # Load the dataset
    dataset = load_dataset(dataset_path)[split]

    ########### Choose the prompt head based on the options ###########
    if num_shots and explicit_errors:
        prompt_head = few_shots_explicit
    elif num_shots and not explicit_errors:
        prompt_head = few_shots_no_explicit
    elif not num_shots and explicit_errors:
        prompt_head = no_few_shots_explicit
    else:
        prompt_head = no_few_shots_no_explicit

    dataset = dataset.map(generate_prompt, fn_kwargs={"field": field, \
        "prompt_head": prompt_head, "train": split == 'train', \
        'num_shots': num_shots,'dataset':dataset , 'explicit_errors': explicit_errors})
        
    return dataset

no_shot_no_explicit_val_dataset = get_dataset(
    dataset_path = "boda/kaneko_data",
    split='test',
    field='prompt',
    num_shots=0,
    explicit_errors = 0).select([0,7,77,54,15,48,55,100,97,31])

no_shot_explicit_val_dataset = get_dataset(
    dataset_path = "boda/kaneko_data",
    split='test',
    field='prompt',
    num_shots=0,
    explicit_errors = 1).select([0,7,77,54,15,48,55,100,97,31])

five_shot_no_explicit_val_dataset = get_dataset(
    dataset_path = "boda/kaneko_data",
    split='test',
    field='prompt',
    num_shots=5,
    explicit_errors = 0).select([0,7,77,54,15,48,55,100,97,31])

five_shot_explicit_val_dataset = get_dataset(
    dataset_path = "boda/kaneko_data",
    split='test',
    field='prompt',
    num_shots=0,
    explicit_errors = 0).select([0,7,77,54,15,48,55,100,97,31])


import json
configs = []
with open('evaluation_sheet.json') as json_file:
   configs = json.load(json_file)

from os import walk

done = []
for (dirpath, dirnames, filenames) in walk('human_eval'):
    for fn in filenames:
        done.append(fn.split('.')[0])

def write_output(filename,predicitons, labels, Errors, corrects ):
    with open(filename, 'w') as f:
        for i in range(len(predicitons)):
            f.write(f"Error_sentence: {Errors[i]}\n\n")
            f.write(f"Correct_sentence: {corrects[i]}\n\n")
            f.write(f"Prediction: {predicitons[i]}\n\n")
            f.write(f"Label: {labels[i]}\n\n")
            f.write(f'-' * 30)
            f.write('\n\n\n')
            


def inference(prompts, tokenizer, model):
    
   
    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0001,
            repetition_penalty = 5.0,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            num_beams=10,
            # generation_config=generation_config,
        )
    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

    return output_text
        

def inf(val_dataloader,c):
    
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
    model = AutoModelForCausalLM.from_pretrained(
    c['model_name'],
    quantization_config=bnb_config,
    trust_remote_code=True,
    # use_flash_attention_2=model_args.use_flash_attention_2,
    use_flash_attention_2=0,

    )

    if c['chkpt']:
        print(f"Loading model from {c['model_name']}")
        adapter_checkpoint  = c['chkpt']
        model = PeftModel.from_pretrained(model, adapter_checkpoint)

    else:
        print(f"Loading Base Model {c['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(c['model_name'])
    model = model.eval()
    # Define PAD Token = BOS Token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id
    predicitons = []
    labels = []
    error_sentences = []
    correct_sentences = []
    
    for batch in tqdm(val_dataloader):

        prompts = [x['prompt'] for x in batch]
        for i,ex in enumerate(batch):
            # labels.append(f"{i+1}. {ex['explanation']}\n")
            labels.append(ex['label'])
            error_sentences.append(ex['incorrect_sentence'])
            correct_sentences.append(ex['correct_sentence'])

        output_text = inference(prompts=prompts, tokenizer=tokenizer, model=model)

        predicitons.extend(output_text)

    filename = f"human_eval/{c['id']}.txt"

    
    assert len(predicitons) == len(labels) == len(error_sentences) == len(correct_sentences)
    write_output(filename,predicitons, labels, error_sentences, correct_sentences)

for c in configs:
    print(c['model_name'])

import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'




for c in configs:

    if str(c['id']) in done:
        continue

    # if c['model_name'] == 'cognitivecomputations/dolphin-2.8-mistral-7b-v02':
    #     continue
    # torch.cuda.empty_cache()

    print(f'----- Generating for {c["model_name"]} -----')
    
    if c['shots'] == 0 and c['explicit_errors'] == 0:
        val_dataset = no_shot_no_explicit_val_dataset
    elif c['shots'] == 0 and c['explicit_errors'] == 1:
        val_dataset = no_shot_explicit_val_dataset
    elif c['shots'] == 5 and c['explicit_errors'] == 0:
        val_dataset = five_shot_no_explicit_val_dataset
    elif c['shots'] == 5 and c['explicit_errors'] == 1:
        val_dataset = five_shot_explicit_val_dataset

    val_dataloader = DataLoader(val_dataset, batch_size=2,collate_fn=lambda x: x )

    ## chatgpt inference
    if c['model_name'] == 'chatgpt':
        continue
    elif 'Meta' in c['model_name']:
        continue
    ## Mistral and mixtral inference
    else:
        inf(val_dataloader,c)

    done.append(str(c['id']))