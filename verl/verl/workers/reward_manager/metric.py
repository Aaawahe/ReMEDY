# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import torch
import numpy as np
from collections import defaultdict

from verl import DataProto

def calculate_ngram_overlap(response_token_list, ngram_num=3):
    response_ngram_list = []
    for response_token in response_token_list:
        response_ngram = []
        for i in range(len(response_token) - ngram_num + 1):
            response_ngram.append(tuple(response_token[i:i+ngram_num]))
        response_ngram_list.append(response_ngram)
    ngram_overlap_list = []
    for i in range(len(response_ngram_list)):
        for j in range(i+1, len(response_ngram_list)):
            ngram_overlap = len(set(response_ngram_list[i]) & set(response_ngram_list[j])) * 1.0 / len(set(response_ngram_list[i]) | set(response_ngram_list[j]))
            ngram_overlap_list.append(ngram_overlap)
    return sum(ngram_overlap_list) * 1.0 / len(ngram_overlap_list)

def metric_experience(tokenizer, data: DataProto, step: int = 0):

    # batched scoring
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = data.batch['attention_mask'][:,:prompt_length].sum(dim=-1)

    response_ids = data.batch['responses']
    valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(dim=-1)

    token_level_scores = data.batch['token_level_scores']

    # data_source
    data_sources = data.non_tensor_batch.get('data_source')
    prompt_str_to_data_source = {}

    response_token_by_prompt_str = defaultdict(list)
    # response score
    response_score_by_prompt_str = defaultdict(list)
    # response correct
    response_correct_by_prompt_str = defaultdict(list)
    # response length when correct
    response_length_when_correct = []
    # response length when wrong
    response_length_when_wrong = []

    for i in range(len(data)):
        valid_prompt_ids = prompt_ids[i][-valid_prompt_length[i]:]
        valid_response_ids = response_ids[i][:valid_response_length[i]]
        prompt_str = tokenizer.decode(valid_prompt_ids)
        response_str = tokenizer.decode(valid_response_ids)

        prompt_str_to_data_source[prompt_str] = data_sources[i]
        response_token_by_prompt_str[prompt_str].append(valid_response_ids.tolist())

        # score > 0.90 表示正确
        if float(token_level_scores[i][valid_response_length[i].item()-1]) > 0.90:
            response_correct_by_prompt_str[prompt_str].append(1.0)
            response_length_when_correct.append(float(valid_response_length[i].item()))
        else:
            response_correct_by_prompt_str[prompt_str].append(0.0)
            response_length_when_wrong.append(float(valid_response_length[i].item()))

        # response score
        response_score_by_prompt_str[prompt_str].append(float(token_level_scores[i][valid_response_length[i].item()-1]))

    # response pass n
    response_pass_n_by_prompt_str = {k: 1.0 if max(v) == 1.0 else 0.0 for k, v in response_correct_by_prompt_str.items()}

    # response accuracy
    response_accuracy_by_prompt_str = {k: np.mean([1.0 if x == 1.0 else 0.0 for x in v]) for k, v in response_correct_by_prompt_str.items()}

    response_mean_score_by_prompt_str = {k: np.mean(v) for k, v in response_score_by_prompt_str.items()}
    response_non_zero_diff_count_by_prompt_str = {k: sum([1.0 if abs(x - response_mean_score_by_prompt_str[k]) > 0.1 else 0.0 for x in response_score_by_prompt_str[k]]) for k in response_score_by_prompt_str.keys()}
    
    result = {
        'response/pass_n': 
            np.mean(list(response_pass_n_by_prompt_str.values())),
        'response/accuracy': 
            np.mean(list(response_accuracy_by_prompt_str.values())),
        'response_length/mean_when_correct': 
            np.mean(response_length_when_correct) if len(response_length_when_correct) > 0 else 0.0,
        'response_length/mean_when_wrong':
            np.mean(response_length_when_wrong) if len(response_length_when_wrong) > 0 else 0.0,
    }

    response_zero_adv_count_by_data_source = defaultdict(int)
    for prompt_str, count in response_non_zero_diff_count_by_prompt_str.items():
        if count <= 0:
            if prompt_str in prompt_str_to_data_source:
                response_zero_adv_count_by_data_source[prompt_str_to_data_source[prompt_str]] += 1
            response_zero_adv_count_by_data_source["ALL"] += 1
    
    for data_source, count in response_zero_adv_count_by_data_source.items():
        result[f'response/zero_adv_count/{data_source}'] = count
            
    # TODO Maybe slow if too many rollouts in one group
    if step == 0 or step % 8 == 0:
        response_ngram_overlap_by_prompt_str = {k: calculate_ngram_overlap(v) for k, v in response_token_by_prompt_str.items()}
        result['response/ngram_overlap_score'] = np.mean(list(response_ngram_overlap_by_prompt_str.values()))
    
    return result