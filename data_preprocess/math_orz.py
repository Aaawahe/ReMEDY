import re
import os
import sys
import json
import random
from tqdm import tqdm
import pandas
import pyarrow.parquet as pq
import pyarrow as pa

from prompt import *

MATH_PROMPT = [
    """
Answer the math problem below, your final answer will be extracted automatically by the \\boxed{{}} tag.
This is the problem:
{prompt}
    """.strip(),

    """
put your final answer within \\boxed{{}}.
This is the problem:
{prompt}
    """.strip(),

    """
Please reason step by step, and put your final answer within \\boxed{{}}.
This is the problem:
{prompt}
    """.strip(),
]

if __name__ == '__main__':

    data_source = "orz_math"

    path = "/data/Open-Reasoner-Zero/orz_math_57k_collected.json"
    output_path = f"/data/train/{data_source}.parquet"
    
    new_data = []
    data = json.loads(open(path, "r", encoding="utf-8").read())
    for item in tqdm(data):
        prompt = random.choice(MATH_PROMPT).format(prompt=item[0]["value"])
        new_item = {
            "data_source": data_source,
            "reward_actor": "RewardActorMath",
            "prompt": R1_ORIGIN_PROMPT_ADD_LANGUAGE.format(prompt=prompt),
            "reward_model": {
                "style": "rule",
                "ground_truth": item[1]["ground_truth"]["value"]
            },
            "extra_info": {
                "question": item[0]["value"],
                "ground_truth": item[1]["ground_truth"]["value"],
                "index": len(new_data),
            }
        }
        new_data.append(new_item)
    
    # to parquet
    df = pandas.DataFrame(new_data)
    print(df)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
