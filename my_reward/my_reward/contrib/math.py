import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from my_reward.utils.time_utils import timeprint
from my_reward.auxiliary.format_reward import (
    get_think_and_answer
)
from my_reward.auxiliary.math_utils import (
    is_equal,
    solution2answer
)
from my_reward.contrib.base import RewardActorBase

async def get_task(tasks):
    return await asyncio.gather(*tasks)
    
class RewardActorMath(RewardActorBase):
    
    @classmethod
    def batch_compute_score(
        cls, 
        params, 
        data_source_list, 
        prompt_str_list, 
        response_str_list, 
        ground_truth_list, 
        extra_info_list, 
        finish_reason_list=None
    ):
        
        if finish_reason_list is None:
            finish_reason_list = [None] * len(prompt_str_list)

        result = []
        for _ in response_str_list:
            result.append({
                "reason": "DEFAULT",
                "reward": cls.default
            })

        pattern = re.compile(r".*?(\\boxed{.*}).*?", re.DOTALL)

        skip_index = set()
        index_list = []
        extracted_answer_list = []

        for i, (prompt_str, response_str, ground_truth, finish_reason) in enumerate(zip(
            prompt_str_list, response_str_list, ground_truth_list, finish_reason_list)):

            format_score = cls.compute_format_score(prompt_str, response_str, finish_reason)
            if format_score != 1.0:
                result[i] = {
                    "reason": "FORMAT_WRONG",
                    "reward": format_score / 10.0
                }
                skip_index.add(i)
                continue

            _, answer_str = get_think_and_answer(response_str)
            _match = re.findall(pattern, answer_str)
            extracted_answer = _match[-1] if _match else ""
            if not extracted_answer:
                result[i] = {
                    "reason": "FORMAT_WRONG",
                    "reward": 0.0
                }
                skip_index.add(i)
                continue
            extracted_answer_list.append(extracted_answer)
            index_list.append(i)
        
        stt = time.time()
        executor = ThreadPoolExecutor(max_workers=64)
        equal_tasks = []
        for index, extracted_answer in zip(index_list, extracted_answer_list):
            equal_tasks.append(is_equal(solution2answer(ground_truth_list[index]), solution2answer(extracted_answer), executor, math_mode="math_verify"))
        equal_results = asyncio.run(get_task(equal_tasks))
        edt = time.time()
        timeprint(f"####### math is_equal time: {edt - stt} s for {len(index_list)} items")

        timeout_count = 0
        for index, equal_result in zip(index_list, equal_results):
            equal_flag, timeout_flag = equal_result
            if equal_flag:
                result[index] = {
                    "reason": "CORRECT",
                    "reward": 1.0
                }
            elif timeout_flag:
                result[index] = {
                    "reason": "TIMEOUT",
                    "reward": 0.1
                }
                timeout_count += 1
            else:
                result[index] = {
                    "reason": "WRONG",
                    "reward": 0.1
                }
        timeprint(f"####### math timeout count: {timeout_count} / {len(index_list)}")
        return cls.add_penalty(result, prompt_str_list, response_str_list, extra_info_list)

        