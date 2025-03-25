import re
import os
import sys
import json
from tqdm import tqdm
import pandas
import pyarrow.parquet as pq
import pyarrow as pa

from prompt import *

if __name__ == '__main__':

    data_source = "medmcqa"

    path = "/data/medmcqa/train_data.jsonl"
    output_path = f"/data/train/{data_source}.parquet"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    new_data = []
    for line in tqdm(open(path, "r", encoding="utf-8").readlines()):
        line = line.strip()
        item = json.loads(line)

        new_item = {
            "data_source": data_source,
            "reward_actor": "RewardActorMCQA",
            "prompt": R1_ORIGIN_PROMPT_ADD_LANGUAGE.format(prompt=item["reformat_question"]),
            "reward_model": {
                "style": "rule",
                "ground_truth": item["options"][item["label"]]
            },
            "extra_info": {
                "original_question": item["question"],
                "question": item["reformat_question"], # 必须存在，用来 verify
                "options": item["options"],
                "index": len(new_data),
            }
        }
        new_data.append(new_item)
    
    # to parquet
    df = pandas.DataFrame(new_data)
    print(df)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
