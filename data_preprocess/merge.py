import re
import os
import json
from tqdm import tqdm
import pandas
import pyarrow.parquet as pq
import pyarrow as pa

DATA = [
    [
        "/data/train/orz_math.parquet",
        6348,
    ],
    [
        "/data/train/medqa.parquet",
        4146,
    ],
    [
        "/data/train/medmcqa.parquet",
        6348,
    ],

]

OUTPUT_PATH = "/data/train/merge.parquet"

def merge_parquet(data, output_path):
    dfs = []
    for path, limit in tqdm(data):
        df = pq.read_table(path).to_pandas()
        size = len(df)        
        while size < limit:
            dfs.append(df)
            limit -= size
        df = df.sample(n=limit)
        dfs.append(df)
    merged_df = pandas.concat(dfs)
    table = pa.Table.from_pandas(merged_df)
    print(merged_df)
    pq.write_table(table, output_path)

if __name__ == '__main__':
    merge_parquet(DATA, OUTPUT_PATH)
