import os
import argparse

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str)
    parser.add_argument("--checkpoint_inputpath", type=str)
    parser.add_argument("--checkpoint_outputpath", type=str)
    parser.add_argument("--world_size", type=int)
    args = parser.parse_args()

    step = args.step
    checkpoint_inputpath = args.checkpoint_inputpath
    checkpoint_outputpath = args.checkpoint_outputpath
    world_size = args.world_size

    state_dict = defaultdict(list)

    for rank in range(world_size):
        filepath = f"{checkpoint_inputpath}/global_step_{step}/actor/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    os.makedirs(f"{checkpoint_outputpath}/global_step_{step}", exist_ok=True)

    config = AutoConfig.from_pretrained(f"{checkpoint_inputpath}/global_step_{step}/actor/huggingface")
    config.save_pretrained(f"{checkpoint_outputpath}/global_step_{step}")

    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(f"{checkpoint_outputpath}/global_step_{step}", max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_inputpath}/global_step_{step}/actor/huggingface")
    tokenizer.save_pretrained(f"{checkpoint_outputpath}/global_step_{step}")
    
if __name__ == "__main__":
    main()