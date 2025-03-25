import re
import math
from my_reward.utils.time_utils import timeprint
from my_reward.auxiliary.format_reward import (
    score_think_pattern,
    endswith_think, 
    get_think_and_answer
)
from my_reward.auxiliary.language_reward import (
    score_language_consistency,
)

class RewardActorBase:

    default = 0.0

    @classmethod
    def compute_format_score(
        cls, 
        prompt, 
        response, 
        finish_reason=None
    ):
        result = score_think_pattern(
            response, 
            not_need_think_at_start=endswith_think(prompt), 
            not_need_answer_tag=("<answer>" not in prompt),
            overlong=(finish_reason == "length"))
        return float(result)

    @classmethod
    def compute_language_score(
        cls, 
        response,
        prompt
    ):
        return score_language_consistency(prompt, response)

    @classmethod
    def compute_think_length_score(
        cls, 
        response
    ):
        """
        思考长度相对答案长度，越长得分越高，大于 2 倍以上 clip
        """
        think_str, answer_str = get_think_and_answer(response)
        think_str_length = len(think_str)
        answer_str_length = len(answer_str)
        if answer_str_length == 0:
            return 0.0
        # return (1.0 - math.exp(- min(think_str_length / answer_str_length, 2))) / (1.0 - math.exp(-2))
        return (1.0 - math.exp(- min(think_str_length / answer_str_length, 2))) / 0.8646647167

    @classmethod
    def add_penalty(
        cls, 
        result, 
        prompt_str_list, 
        response_str_list, 
        extra_info_list
    ):
        timeprint(f"### start calculating penalty")
        for i, (prompt_str, response_str, extra_info) in enumerate(zip(
            prompt_str_list, response_str_list, extra_info_list)):
            question = extra_info["question"]
            think_str, answer_str = get_think_and_answer(response_str)
            think_language_score = cls.compute_language_score(response=think_str, prompt=question)
            answer_language_score = cls.compute_language_score(response=answer_str, prompt=question)
            result[i]["reward"] -= (1.0 - think_language_score) / 10.0
            result[i]["reward"] -= (1.0 - answer_language_score) / 10.0
            think_length_score = cls.compute_think_length_score(response_str)
            result[i]["reward"] -= (1.0 - think_length_score) / 10.0
        timeprint(f"### end calculating penalty")
        return result