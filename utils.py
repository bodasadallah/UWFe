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


def extract_output(batch):    
    outputs = []
    for j,sample in enumerate(batch):

        print(sample)
        # lines = sample.split('\n')
        # for i,l in enumerate(lines):

        #     if l=='### Response:':
        #         outputs.append( lines[i+1].lower().strip() )
        #         break
        # if len(outputs) <= j :
        #     outputs.append('')
    
    return outputs
            
