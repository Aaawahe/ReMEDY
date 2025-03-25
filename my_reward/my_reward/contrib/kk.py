import re
import time
from typing import Optional, Dict

from my_reward.auxiliary.format_reward import (
    get_think_and_answer
)
from my_reward.contrib.base import RewardActorBase

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    # print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            # print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    # print("\n[Model Answer Parsing]")
    # print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    # print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        # print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            # print(f"  Found: {name} → {role}")
        else:
            # print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict
    
class RewardActorKK(RewardActorBase):
    
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

        for i, (prompt_str, response_str, ground_truth, finish_reason) in enumerate(zip(
            prompt_str_list, response_str_list, ground_truth_list, finish_reason_list)):

            format_score = cls.compute_format_score(prompt_str, response_str, finish_reason)
            if format_score != 1.0:
                result[i] = {
                    "reason": "FORMAT_WRONG",
                    "reward": format_score / 10.0
                }
                continue

            gt_status = parse_solution_text_format(ground_truth)
            expected_names = list(gt_status.keys())
            # print(f"[Ground Truth] Final identities: {gt_status}")

            # Extract model answer
            _, answer_str = get_think_and_answer(response_str)
            pred_status = parse_model_answer(answer_str, expected_names)
            if pred_status:
                # print(f"\n[Content Validation]")
                # print(f"  Expected: {gt_status}")
                # print(f"  Predicted: {pred_status}")
                if pred_status == gt_status:
                    result[i] = {
                        "reason": "CORRECT",
                        "reward": 1.0
                    }
                else:
                    result[i] = {
                        "reason": "WRONG",
                        "reward": 0.1
                    }
            else:
                result[i] = {
                    "reason": "FORMAT_WRONG",
                    "reward": 0.05
                }

        return cls.add_penalty(result, prompt_str_list, response_str_list, extra_info_list)

        