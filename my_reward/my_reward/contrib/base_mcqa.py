VERIFY_SYSTEM_PROMPT_ZH = """## 角色
请你作为老师判断学生的回答是否为问题的答案

## 任务
请你基于 (1) 问题 (2) 学生的回答 (3) 问题的正确答案 (4) 问题的误导选项，即一些错误答案的候选
判断学生的回答是与正确答案相符，评为 0/1/2 分
2分答案标准:
最终回答与正确答案意思完全一致，允许是正确答案的同义词、缩写等，但不能够包含错误选项
1分答案标准:
最终回答与正确答案意思一致，允许是正确答案的同义词、缩写、子集等，允许在这基础上有一定补充，但不能够包含错误选项
0分答案标准:
最终回答与正确答案不一致，或者包含错误选项，或者包含乱码、格式错误、乱序、多余无关信息
请在给出评分原因后，输出具体分数

## 输入
{{
    "question": "问题",
    "answer": "学生回答",
    "correct_answer": "正确答案",
    "misleading_options": ["误导选项1", "误导选项2", ...]
}}

## 输出格式
请按照下面的json格式返回
```json
{{
    "reason": "评分原因",
    "score": 0/1/2
}}
```
""".strip()

VERIFY_SYSTEM_PROMPT_EN = """## Role
Please judge whether the student's answer is the answer to the question as a teacher

## Task
Based on (1) the question (2) the student's answer (3) the correct answer to the question (4) misleading options of the question, i.e., some wrong answer candidates
Judge the student's answer is consistent with the correct answer, and give a score of 0/1/2
2-point answer criteria:
The final answer is completely consistent with the correct answer, allowing synonyms, abbreviations, etc., but not including wrong options
1-point answer criteria:
The final answer is consistent with the correct answer, allowing synonyms, abbreviations, subsets, etc., and allowing some supplements on this basis, but not including wrong options
0-point answer criteria:
The final answer is inconsistent with the correct answer, or contains wrong options, or contains garbled characters, format errors, disorder, or irrelevant information
Please output the specific score after giving the scoring reason

## Input
{{
    "question": "question",
    "answer": "student's answer",
    "correct_answer": "correct answer",
    "misleading_options": ["misleading option 1", "misleading option 2", ...]
}}

## Output Format
Please return in the json format below
```json
{{
    "reason": "scoring reason",
    "score": 0/1/2
}}
```
""".strip()

VERIFY_USER_PROMPT_ZH = """## 问题:
{question}
## 学生回答:
{answer}
## 正确答案:
{correct_answer}
## 误导选项:
{misleading_options}

## 请给出评分原因后，输出具体分数，以json格式输出
""".strip()

VERIFY_USER_PROMPT_EN = """## Question:
{question}
## Student's Answer:
{answer}
## Correct Answer:
{correct_answer}
## Misleading Options:
{misleading_options}
""".strip()

import re
import time
from my_reward.utils.time_utils import timeprint
from my_reward.api import oneapi_post_by_langchain, read_json
from my_reward.auxiliary.format_reward import (
    get_think_and_answer
)
from my_reward.contrib.base import RewardActorBase
from pydantic import BaseModel, Field

class Score(BaseModel):
    reason: str = Field(..., title="Scoring Reason", description="Scoring Reason")
    score: float = Field(..., title="Score", description="Score")

class RewardActorMCQA(RewardActorBase):
    
    @classmethod
    def normalize_score(cls, score: float):
        if score >= 2.0:
            return 1.0
        elif score >= 1.0:
            return 0.5
        else:
            return 0.1

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

        skip_index = set()
        for i, (prompt_str, response_str, ground_truth, finish_reason) in enumerate(zip(
            prompt_str_list, response_str_list, ground_truth_list, finish_reason_list)):

            format_score = cls.compute_format_score(prompt_str, response_str, finish_reason)
            if format_score != 1.0:
                result[i] = {
                    "reason": "FORMAT_WRONG",
                    "reward": format_score / 10.0
                }
                skip_index.add(i)

        system_prompt = VERIFY_SYSTEM_PROMPT_EN
        index_list = []
        prompt_list = []
        for i, (data_source, prompt_str, response_str, ground_truth, extra_info) in enumerate(zip(data_source_list, prompt_str_list, response_str_list, ground_truth_list, extra_info_list)):
            if i in skip_index:
                continue
            _, answer_str = get_think_and_answer(response_str)
            options = extra_info["options"]
            misleading_options = [v for k, v in options.items() if v != ground_truth]
            prompt = VERIFY_USER_PROMPT_EN.format(
                question=extra_info["question"],
                answer=answer_str,
                correct_answer=ground_truth,
                misleading_options=misleading_options
            )
            prompt_list.append(prompt)
            index_list.append(i)

        batch_size = 256
        response_list = []
        for i in range(0, len(prompt_list), batch_size):
            stt = time.time()
            response_list += oneapi_post_by_langchain(
                prompt=prompt_list[i:i+batch_size],
                system_prompt=system_prompt,
                # base_model=Score,
                **params
            )
            edt = time.time()
            timeprint(f"-------- base mcqa compute score batch size: {len(prompt_list[i:i+batch_size])}, oneapi time: {edt - stt} s")
        
        for index, res in zip(index_list, response_list):
            try:
                res_json = read_json(res)
                result[index] = {
                    "reason": res_json["reason"],
                    "reward": cls.normalize_score(float(res_json["score"]))
                }
            except Exception as e:
                result[index] = {
                    "reason": f"ERROR IN VERIFY: {res}",
                    "reward": cls.default,
                    "exception": str(e)
                }

        return cls.add_penalty(result, prompt_str_list, response_str_list, extra_info_list)
