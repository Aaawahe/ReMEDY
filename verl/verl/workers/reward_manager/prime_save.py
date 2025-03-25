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
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.logging_utils import timestamp

class PrimeSaveRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.save_path = save_path

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data.batch['attention_mask'][:,:prompt_length].sum(dim=-1)

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(dim=-1)
        
        # sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]

        data_sources = data.non_tensor_batch['data_source']
        reward_actors = data.non_tensor_batch['reward_actor']
        
        extra_infos = data.non_tensor_batch['extra_info']

        finish_reasons = data.non_tensor_batch.get('finish_reason')
        if finish_reasons is None:
            finish_reasons = [None for _ in range(len(ground_truth))]

        stop_reasons = data.non_tensor_batch.get('stop_reason')
        if stop_reasons is None:
            stop_reasons = [None for _ in range(len(ground_truth))]
        
        assert len(prompt_ids) == len(response_ids) == len(ground_truth) == len(data_sources)

        datas = []

        for i in range(len(data)):
            valid_prompt_ids = prompt_ids[i][-valid_prompt_length[i]:]
            valid_response_ids = response_ids[i][:valid_response_length[i]]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            datas.append({
                'data_source': data_sources[i],
                'reward_actor': reward_actors[i],
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth[i],
                'extra_info': extra_infos[i],
                'stop_reason': stop_reasons[i],
                'finish_reason': finish_reasons[i]
            })

        stt = time.time()

        try:
            results = self.compute_score(
                actor_list=[x['reward_actor'] for x in datas],
                data_source_list=[x['data_source'] for x in datas],
                prompt_str_list=[x['prompt_str'] for x in datas],
                response_str_list=[x['response_str'] for x in datas],
                ground_truth_list=[x['ground_truth'] for x in datas],
                extra_info_list=[x['extra_info'] for x in datas],
                finish_reason_list=[x['finish_reason'] for x in datas]
            )
            scores = []
            reasons = []
            # Record the experience of `verify` exceptions without performing loss calculation.
            is_errors = []

            if self.save_path is not None:
                if not os.path.exists(os.path.dirname(self.save_path)):
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                with open(self.save_path, 'w', encoding='utf-8') as f:
                    for i in range(len(datas)):
                        result = results[i]
                        
                        reason = ''
                        is_error = False
                        if isinstance(result, list):
                            result = result[0]
                        if isinstance(result, Exception) or result is None:
                            score = 0.0
                        elif isinstance(result, (int, float, bool)):
                            score = float(result)
                        elif isinstance(result, (tuple, list)):
                            score = float(result[0])
                            reason = result[1] if len(result) > 1 else ''
                        elif isinstance(result, dict):
                            score = float(result.get('reward', 0.0))
                            reason = result.get('reason', '')
                            if result.get("exception"):
                                is_error = True
                        else:
                            score = 0.0
                            reason = "unexpected result type"

                        extra_info = datas[i]['extra_info']
                        extra_info = {k: v.tolist() if isinstance(v, (torch.Tensor, np.ndarray)) else v for k, v in extra_info.items()}

                        f.write(json.dumps({
                            'data_source': datas[i]['data_source'],
                            'prompt_str': datas[i]['prompt_str'],
                            'response_str': datas[i]['response_str'],
                            'ground_truth': datas[i]['ground_truth'],
                            'extra_info': extra_info,
                            'finish_reason': finish_reasons[i],
                            'stop_reason': stop_reasons[i],
                            'prompt_str_length': valid_prompt_length[i].item(),
                            'response_str_length': valid_response_length[i].item(),
                            'score': score,
                            'reason': reason,
                            'is_error': is_error
                        }, ensure_ascii=False) + '\n')   
                        
                        scores.append(score)
                        reasons.append(reason)
                        is_errors.append(is_error)

        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(ground_truth))]
            reasons = ['UNEXPECTED ERROR' for _ in range(len(ground_truth))]
            is_errors = [True for _ in range(len(ground_truth))]

        edt = time.time()
        # print(f"{timestamp()} ############## {len(data)} samples Reward computation time: {edt - stt:.2f}s")
        # print(f"{timestamp()} ############## {len(data)} samples with errors: {sum(is_errors)}")

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"#### data_source: {datas[i]['data_source']}\n" \
                      f"#### reward_actor: {datas[i]['reward_actor']}\n" \
                      f"#### prompt: {datas[i]['prompt_str']}\n" \
                      f"#### response: {datas[i]['response_str']}\n" \
                      f"#### response_length: {float(valid_response_length[i].item())}\n" \
                      f"#### ground_truth: {datas[i]['ground_truth']}\n" \
                      f"#### finish_reason: {finish_reasons[i]}\n" \
                      f"#### stop_reason: {stop_reasons[i]}\n" \
                      f"#### score: {scores[i]}\n" \
                      f"#### reason: {reasons[i]}")

        return reward_tensor
